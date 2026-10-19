import os
import shutil
from pathlib import Path
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from wandb.sdk.lib import paths


def mkdir(folder: Path):
    if isinstance(folder, Path):
        folder.mkdir(parents=True, exist_ok=True)
    else:
        os.makedirs(folder, exist_ok=True)

class Checkpointer:
    def __init__(self, root_dir, max_chkpts: int = 0):
        self.root_dir = root_dir
        self.max_chkpts = max_chkpts
        self.history = {}

        self.is_init = dist.is_initialized()
        self.is_master = dist.get_rank() == 0

        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self):
        """Populate history from disk, so resuming training respects max_chkpts."""""
        for f in os.listdir(self.cfg.output_folder):
            if f.startswith("config-") and f.endswith(".pt"):
                step_str = f.split("-")[-1].split(".")[0]
                step = int(step_str)

                paths = [
                    os.path.join(self.root_dir, f"config-{step_str}.pt"),
                    os.path.join(self.root_dir, f"reader-{step_str}.pt"),
                ]

                if not os.path.isdir(os.path.join(self.root_dir, f"model-{step_str}")): # FSDP2 saves states in folders by default
                    ext = ""
                else: # DDP saves states in files with ".pt" extension
                    ext = ".pt"

                paths.extend([
                    os.path.join(self.root_dir, f"model-{step_str}{ext}"),
                    os.path.join(self.root_dir, f"optimizer-{step_str}{ext}"),
                ])

                self.history[step] = paths

    def _trim_chkpts(self):
        if self.max_chkpts <= 0:
            return

        while len(self.history) > self.max_chkpts:
            oldest_step = min(self.history.keys())
            oldest_paths = self.history.pop(oldest_step)

            if self.is_master:
                for p in oldest_paths:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    elif os.path.isfile(p):
                        os.remove(p)

    def save(self, step: int, backend_type: str, config, reader, model, multi_optimizer):
        assert backend_type in ["single", "ddp", "fsdp2"]
        step_str = f"{step:08d}"

        paths = {
            "config": os.path.join(self.root_dir, f"config-{step_str}.pt"),
            "reader": os.path.join(self.root_dir, f"reader-{step_str}.pt"),
        }

        if backend_type == "fsdp2":
            paths["model"] = os.path.join(self.root_dir, f"model-{step_str}")
            paths["optim"] = os.path.join(self.root_dir, f"optimizer-{step_str}")

            # all ranks are involved in dcp
            model_sd, optim_sd = get_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers))

            dcp.save({"model": model_sd}, checkpoint_id=paths["model"])
            dcp.save({"optimizer": optim_sd}, checkpoint_id=paths["optim"])

            if self.is_master:
                torch.save(config, paths["config"])
                torch.save(reader, paths["reader"])
            # end if
        else: # "single" or "ddp"
            paths["model"] = os.path.join(self.root_dir, f"model-{step_str}.pt")
            paths["optim"] = os.path.join(self.root_dir, f"optimizer-{step_str}.pt")

            if self.is_master:
                unwrapped = model.module if hasattr(model, "module") else model
                torch.save(obj=unwrapped.state_dict(), f=paths["model"])
                torch.save(obj=[opt.state_dict() for opt in multi_optimizer.optimizers], f=paths["optim"])
                torch.save(obj=reader, f=paths["reader"])
                torch.save(obj=config, f=paths["config"])
            # end if
        # end if-else

        if self.is_init:
            dist.barrier()

        self.history[step] = paths
        self._trim_chkpts()

    def load(self, step: int, backend_type: str, model, multi_optimizer):
        assert backend_type in ["single", "ddp", "fsdp2"]
        step_str = f"{step:08d}"

        if backend_type == "fsdp2":
            model_dir = os.path.join(self.root_dir, f"model-{step_str}")
            optim_dir = os.path.join(self.root_dir, f"optimizer-{step_str}")

            model_sd, optim_sd = get_state_dict(model=model, optimizers=tuple(multi_optimizer.optimizers))

            dcp.load({"model": model_sd}, checkpoint_id=model_dir) # inplace loading
            dcp.load({"optimizer": optim_sd}, checkpoint_id=optim_dir) # inplace loading

            set_state_dict(model, tuple(multi_optimizer.optimizers), model_state_dict=model_sd, optim_state_dict=optim_sd)
        else:
            device = f"cuda:{dist.get_rank()}" if self.is_init else "cuda"

            unwrapped = model.module if hasattr(model, "module") else model

            model_state = torch.load(os.path.join(self.root_dir, f"model-{step_str}.pt"))
            optim_states = torch.load(os.path.join(self.root_dir, f"optimizer-{step_str}.pt"))

            unwrapped.load_state_dict(model_state, map_location=device, weights_only=False)
            for opt, sd in zip(multi_optimizer.optimizers, optim_states):
                opt.load_state_dict(sd, map_location=device, weights_only=False)
        # end if-else

        config_path = os.path.join(self.root_dir, f"config-{step_str}.pt")
        reader_path = os.path.join(self.root_dir, f"reader-{step_str}.pt")

        config = torch.load(config_path, map_location="cpu", weights_only=False)
        reader = torch.load(reader_path, map_location="cpu", weights_only=False)

        return config, reader
