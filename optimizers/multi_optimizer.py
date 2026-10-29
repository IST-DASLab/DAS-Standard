import torch
import torch.nn as nn
from torch.optim import AdamW
from argparse import Namespace as ArgParseNamespace
from bitsandbytes.optim import AdamW8bit

from backends import BaseBackend

from .muon import MuonSingle, MuonDDP, MuonSingleBlock, MuonDDPBlock, MuonFSDP2All2All as MuonFSDP2, MuonFSDP2Block
from .lr_scheduler import LRScheduler

class MultiOptimizer:
    def __init__(self, cfg: ArgParseNamespace, model: nn.Module, backend: BaseBackend):
        self.cfg = cfg
        self.model = model
        self.backend = backend

        self.lr_scheduler = LRScheduler(cfg)

        self.optim_iter = 0
        self.optimizers = self._create_optimizers_list()
        # self._check()
        self._set_lr_init()

        self.zero_grad(set_to_none=True)

    # def _check(self):
    #     len_set_all_params = len(set(self.model.parameters()))
    #     len_set_optimizers = len(set(
    #         p
    #         for opt in self.optimizers
    #         for group in opt.param_groups
    #         for p in group["params"]
    #     ))
    #
    #     print(f"{len_set_all_params=}; {len_set_optimizers=}")
    #     if self.cfg.wandb.group.startswith("test"):
    #         exit(666)

    def _unwrap_model(self):
        model = self.model
        # Unwrap torch.compile wrapper
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        # Unwrap DDP wrapper
        if hasattr(model, "module"):
            model = model.module
        return model

    def _get_linear_layers(self):
        return [p for p in self._unwrap_model().blocks.parameters() if p.ndim == 2 and p.requires_grad]

    def _separate_params_into_2d_linear_vs_others(self):
        linear_params = self._get_linear_layers()
        linear_ids = {id(p) for p in linear_params}
        other_params = [p for p in self.model.parameters() if id(p) not in linear_ids and p.requires_grad]
        return linear_params, other_params

    def _set_lr_init(self):
        """
        This function is called after creating the variable self.optimizers and creates the variable "lr_init".
        Having "lr_init" in place, we can easily change the learning rate using the scheduler defined in the config.
        """
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr_init"] = group["lr"]

    def _update_learning_rates(self):
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = self.lr_scheduler(t=self.optim_iter, peak=group["lr_init"])

    def extract_learning_rates(self):
        lrs = {}
        for opt in self.optimizers:
            opt_name = opt.__class__.__name__
            lrs[f"lrs/{opt_name}"] = opt.param_groups[0]["lr"]
        return lrs

    def get_optimizer_stats(self):
        stats = {}
        for opt in self.optimizers:
            if hasattr(opt, "logging_data"):
                stats = {**stats, **opt.logging_data}
        return stats

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.optim_iter += 1
        self._update_learning_rates()
        for opt in self.optimizers:
            opt.step()

        return loss

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def _create_optimizers_list(self):
        oc = self.cfg.optimizer # oc = optimizer config

        match oc.optimizer_name:
            case "adamw8b":
                return [
                    AdamW8bit(
                        params=self.model.parameters(),
                        lr=oc.lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps),
                ]
            case "adamw":
                return [
                    AdamW(
                        params=self.model.parameters(),
                        lr=oc.lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps),
                ]
            case "muon":
                muon_fn = None
                match self.cfg.training.backend:
                    case "single":
                        muon_fn = MuonSingle
                    case "ddp":
                        muon_fn = MuonDDP
                    case "fsdp2":
                        muon_fn = MuonFSDP2

                assert muon_fn is not None

                muon_params, adamw_params = self._separate_params_into_2d_linear_vs_others()
                return [
                    muon_fn(
                        params=muon_params,
                        lr=oc.lr,
                        coeffs_type=oc.muon_coeffs_type,
                        scaling_type=oc.muon_scaling_type,
                        weight_decay=oc.weight_decay,
                        momentum=oc.muon_momentum
                    ),
                    AdamW(
                        params=adamw_params,
                        lr=oc.lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps,
                    )
                ]
            case "block_muon":
                block_muon_fn = None
                match self.cfg.training.backend:
                    case "single":
                        block_muon_fn = MuonSingleBlock
                    case "ddp":
                        block_muon_fn = MuonDDPBlock
                    case "fsdp2":
                        block_muon_fn = MuonFSDP2Block

                assert block_muon_fn is not None

                muon_params, adamw_params = self._separate_params_into_2d_linear_vs_others()
                return [
                    block_muon_fn(
                        params=muon_params,
                        lr=oc.lr,
                        num_blocks=oc.muon_num_blocks,
                        coeffs_type=oc.muon_coeffs_type,
                        scaling_type=oc.muon_scaling_type,
                        weight_decay=oc.weight_decay,
                        momentum=oc.muon_momentum
                    ),
                    AdamW(
                        params=adamw_params,
                        lr=oc.lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps,
                    )
                ]
            case _:
                raise NotImplementedError(f'Optimizer {oc.optimizer_name} is not implemented!')