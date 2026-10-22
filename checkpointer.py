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

class CheckpointManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.checkpointer = Checkpointer(
            load_dir=cfg.checkpointing.resume_folder,
            save_dir=cfg.training.output_folder,
            max_limit=cfg.checkpointing.max_limit,
        )

    def setup_resume(self, model, multi_optimizer, train_reader):
        """
            Loads model and optimizer tensor states while enforcing new CLI runtime configs.
            Returns the starting iteration index (start_iter).
        """
        resume_step = self.cfg.checkpointing.resume_step
        if resume_step is None:
            return 1

        if resume_step == -1:
            if not self.checkpointer.history:
                raise RuntimeError(f"No valid checkpoints found in {self.checkpointer.load_dir} to resume from.")
            resume_step = max(self.checkpointer.history.keys())

        print(f"[Checkpointer] Resuming parameter states from step {resume_step} in {self.checkpointer.load_dir}...")

        disk_config, disk_reader = self.checkpointer.load(
            step=resume_step,
            model=model,
            multi_optimizer=multi_optimizer,
        )

        if disk_reader is not None:
            train_reader.load_state_dict(disk_reader.state_dict())

        self._recompute_optimizer_lrs(multi_optimizer)

        return resume_step + 1

    def _recompute_optimizer_lrs(self, multi_optimizer):
        oc = self.cfg.optimizer

        for opt in multi_optimizer.optimizers:
            for group in opt.param_groups:
                group["weight_decay"] = oc.weight_decay
                group["lr_init"] = oc.lr

        multi_optimizer._update_learning_rates()

    def save_checkpoint(self, step: int, model, multi_optimizer, train_reader):
        self.checkpointer.save(
            step=step,
            config=self.cfg,
            reader=train_reader,
            model=model,
            multi_optimizer=multi_optimizer,
        )

class Checkpointer:
    def __init__(self, load_dir: str, save_dir: str, max_limit: int = 0):
        self.load_dir = Path(load_dir)
        self.save_dir = Path(save_dir)
        self.max_limit = max_limit
        self.history = {}

        self.is_init = dist.is_initialized()
        self.is_master = not self.is_init or dist.get_rank() == 0

        if self.is_master:
            # self.load_dir.mkdir(parents=True, exist_ok=True)
            self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.is_init:
            dist.barrier()

        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self):
        """Populate history from disk, so resuming training respects max_limit."""""
        if not self.load_dir.exists():
            return

        for f in os.listdir(self.load_dir):
            if f.startswith("config-") and f.endswith(".pt"):
                step_str = f.split("-")[-1].split(".")[0]
                step = int(step_str)

                self.history[step] = {
                    # files
                    "config_pt": self.load_dir / f"chkpt-{step_str}" / f"config-{step_str}.pt",
                    "config_json": self.load_dir / f"chkpt-{step_str}" / f"config-{step_str}.json",
                    "reader_pt": self.load_dir / f"chkpt-{step_str}" / f"reader-{step_str}.pt",

                    # folders
                    "model": self.load_dir / f"chkpt-{step_str}" / f"model-{step_str}",
                    "optimizer": self.load_dir / f"chkpt-{step_str}" / f"optimizer-{step_str}",
                }

    def _trim_chkpts(self):
        if self.max_limit <= 0:
            return

        while len(self.history) > self.max_limit:
            oldest_step = min(self.history.keys())
            _ = self.history.pop(oldest_step)

            if self.is_master:
                shutil.rmtree(self.save_dir / f"chkpt-{oldest_step:08d}")

    def save(self, step: int, config, reader, model, multi_optimizer):
        step_str = f"{step:08d}"

        paths = {
            # files
            "config_pt": self.save_dir / f"chkpt-{step_str}" / f"config-{step_str}.pt",
            "config_json": self.save_dir / f"chkpt-{step_str}" / f"config-{step_str}.json",
            "reader_pt": self.save_dir / f"chkpt-{step_str}" / f"reader-{step_str}.pt",

            # folders
            "model": self.save_dir / f"chkpt-{step_str}" / f"model-{step_str}",
            "optimizer": self.save_dir / f"chkpt-{step_str}" / f"optimizer-{step_str}",
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
            "config_pt": self.load_dir / f"chkpt-{step_str}" / f"config-{step_str}.pt",
            # "config_json": self.root_dir /  f"chkpt-{step_str}"f/ "config-{step_str}.json", # not needed
            "reader_pt": self.load_dir / f"chkpt-{step_str}" / f"reader-{step_str}.pt",

            # folders
            "model": self.load_dir / f"chkpt-{step_str}" / f"model-{step_str}",
            "optimizer": self.load_dir / f"chkpt-{step_str}" / f"optimizer-{step_str}",
        }

        model_sd, optim_sd = get_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers))

        print(f'Loading model from {paths["model"]}')
        dcp.load({"model": model_sd}, checkpoint_id=paths["model"])

        print(f'Loading optimizer from {paths["optimizer"]}')
        dcp.load({"optimizer": optim_sd}, checkpoint_id=paths["optimizer"])
        set_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers), model_state_dict=model_sd, optim_state_dict=optim_sd)
        multi_optimizer.optim_iter = step # we need -1 because optim_iter is increased before setting the learning rates

        print(f'Loading pt config from {paths["config_pt"]}')
        config = torch.load(paths["config_pt"], map_location="cpu", weights_only=False)

        print(f'Loading pt reader from {paths["reader_pt"]}')
        reader = torch.load(paths["reader_pt"], map_location="cpu", weights_only=False)

        return config, reader
