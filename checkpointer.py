import os
import json
import shutil
from pathlib import Path
from dataclasses import is_dataclass, asdict

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


def _save_config_json(config, path):
    """Serialize dataclasses, dicts, or custom config objects to JSON."""
    if is_dataclass(config):
        data = asdict(config)
    elif isinstance(config, dict):
        data = config
    elif hasattr(config, "__dict__"):
        data = vars(config)
    else:
        data = {"config": str(config)}

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

class Checkpointer:
    def __init__(self, root_dir, max_chkpts: int = 0):
        self.root_dir = Path(root_dir)
        self.max_chkpts = max_chkpts
        self.history = {}

        self.is_init = dist.is_initialized()
        self.is_master = not self.is_init or dist.get_rank() == 0

        if self.is_master:
            self.root_dir.mkdir(parents=True, exist_ok=True)

        if self.is_init:
            dist.barrier()

        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self):
        """Populate history from disk, so resuming training respects max_chkpts."""""
        if not self.root_dir.exists():
            return

        for f in os.listdir(self.root_dir):
            if f.startswith("config-") and f.endswith(".pt"):
                step_str = f.split("-")[-1].split(".")[0]
                step = int(step_str)

                self.history[step] = {
                    # files
                    "config_pt": self.root_dir / f"config-{step_str}.pt",
                    "config_json": self.root_dir / f"config-{step_str}.json",
                    "reader": self.root_dir / f"reader-{step_str}.pt",

                    # folders
                    "model": self.root_dir / f"model-{step_str}",
                    "optimizer": self.root_dir / f"optimizer-{step_str}",
                }

    def _trim_chkpts(self):
        if self.max_chkpts <= 0:
            return

        while len(self.history) > self.max_chkpts:
            oldest_step = min(self.history.keys())
            oldest_paths = self.history.pop(oldest_step)

            if self.is_master:
                for p in oldest_paths.values():
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                        print(f"Deleted folder {p}")
                    elif os.path.isfile(p):
                        os.remove(p)
                        print(f"Deleted file {p}")

    def save(self, step: int, config, reader, model, multi_optimizer):
        step_str = f"{step:08d}"

        paths = {
            # files
            "config_pt": self.root_dir / f"config-{step_str}.pt",
            "config_json": self.root_dir / f"config-{step_str}.json",
            "reader": self.root_dir / f"reader-{step_str}.pt",

            # folders
            "model": self.root_dir / f"model-{step_str}",
            "optimizer": self.root_dir / f"optimizer-{step_str}",
        }

        model_sd, optim_sd = get_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers))

        print(f'Saving model to {paths["model"]}')
        dcp.save({"model": model_sd}, checkpoint_id=paths["model"])

        print(f'Saving optimizer to {paths["optimizer"]}')
        dcp.save({"optimizer": optim_sd}, checkpoint_id=paths["optimizer"])

        if self.is_master:
            print(f'Saving pt config to {paths["config_pt"]}')
            torch.save(obj=config, f=paths["config_pt"])

            print(f'Saving json config to {paths["config_json"]}')
            _save_config_json(config=config, path=paths["config_json"])

            print(f'Saving pt reader to {paths["reader_pt"]}')
            torch.save(obj=reader, f=paths["reader_pt"])

        if self.is_init:
            dist.barrier()

        self.history[step] = paths
        self._trim_chkpts()

    def load(self, step: int, model, multi_optimizer):
        step_str = f"{step:08d}"

        paths = {
            # files
            "config_pt": self.root_dir / f"config-{step_str}.pt",
            # "config_json": self.root_dir / f"config-{step_str}.json", # not needed
            "reader": self.root_dir / f"reader-{step_str}.pt",

            # folders
            "model": self.root_dir / f"model-{step_str}",
            "optimizer": self.root_dir / f"optimizer-{step_str}",
        }

        model_sd, optim_sd = get_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers))

        print(f'Loading model from {paths["model"]}')
        dcp.load({"model": model_sd}, checkpoint_id=paths["model"])

        print(f'Loading optimizer from {paths["optimizer"]}')
        dcp.load({"optimizer": optim_sd}, checkpoint_id=paths["optimizer"])
        set_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers), model_state_dict=model_sd, optim_state_dict=optim_sd)

        print(f'Loading pt config from {paths["config_pt"]}')
        config = torch.load(paths["config_pt"], map_location="cpu", weights_only=False)

        print(f'Loading pt reader from {paths["reader_pt"]}')
        reader = torch.load(paths["reader_pt"], map_location="cpu", weights_only=False)

        return config, reader
