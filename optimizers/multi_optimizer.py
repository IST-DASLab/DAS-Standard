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
        self._set_lr_init()

        self.zero_grad(set_to_none=True)

    def _unwrap_model(self):
        model = self.model
        # Unwrap torch.compile wrapper
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        # Unwrap DDP wrapper
        if hasattr(model, "module"):
            model = model.module
        return model

    def _get_input_embedding(self):
        return [self._unwrap_model().token_embed.weight]

    def _get_pos_embedding(self):
        return [self._unwrap_model().pos_embed.weight] if hasattr(self.model, "pos_embed") else []

    def _get_output_embedding(self):
        return [self._unwrap_model().lm_head.weight]

    def _get_layer_norms(self):
        return [p for p in self._unwrap_model().parameters() if p.ndim == 1]

    def _get_linear_layers(self):
        return [p for p in self._unwrap_model().blocks.parameters() if p.ndim == 2]

    def _get_all_params(self):
        return self.model.parameters()

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

    def get_optimzier_stats(self):
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
                        params=self._get_all_params(),
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
                        params=self._get_all_params(),
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

                return [
                    muon_fn(
                        params=self._get_linear_layers(),
                        lr=oc.lr,
                        coeffs_type=oc.muon_coeffs_type,
                        scaling_type=oc.muon_scaling_type,
                        weight_decay=oc.weight_decay,
                        momentum=oc.muon_momentum
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
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

                return [
                    block_muon_fn(
                        params=self._get_linear_layers(),
                        lr=oc.lr,
                        num_blocks=oc.muon_num_blocks,
                        coeffs_type=oc.muon_coeffs_type,
                        scaling_type=oc.muon_scaling_type,
                        weight_decay=oc.weight_decay,
                        momentum=oc.muon_momentum
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
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