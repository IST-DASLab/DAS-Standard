import torch
import torch.nn as nn
from torch.optim import AdamW
from argparse import Namespace as ArgParseNamespace

from backends import BaseBackend
from lr_scheduler import LRScheduler

from ista_daslab_optimizers import MicroAdam
from experimental_optimizers import MicroAdamTorch, Trion
from bitsandbytes.optim import AdamW8bit

# from .muon import MuonSingle, MuonDDP, MuonFSDP2


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

    def _get_input_embedding(self):
        return [self.model.token_embed.weight]

    def _get_pos_embedding(self):
        return [self.model.pos_embed.weight] if hasattr(self.model, "pos_embed") else []

    def _get_output_embedding(self):
        return [self.model.lm_head.weight]

    def _get_layer_norms(self):
        return [p for p in self.model.parameters() if p.ndim == 1]

    def _get_linear_layers(self):
        return [p for p in self.model.blocks.parameters() if p.ndim == 2]

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
        cfg = self.cfg
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
                        lr=oc.adamw_lr,
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
                        lr=oc.adamw_lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps),
                ]
            case "muon":
                match cfg.training.backend:
                    case "single":
                        muon_fn = MuonSingle
                    case "ddp":
                        muon_fn = MuonDDP
                    case "fsdp2":
                        muon_fn = MuonFSDP2

                return [
                    muon_fn(
                        params=self._get_linear_layers(),
                        lr=oc.muon_lr,
                        weight_decay=oc.weight_decay,
                        momentum=oc.muon_momentum
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
                        lr=oc.adamw_lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps,
                    )
                ]
            case "microadam-paper":
                # the version that optimizes all layers wih MicroAdam
                return [
                    MicroAdam(
                        params=self.model.parameters(),
                        lr=oc.adamw_lr,
                        m=oc.microadam_m,
                        quant_block_size=oc.microadam_q_block,
                        k_init=oc.microadam_k,
                        alpha=0, # 0 means sparse update
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        weight_decay=oc.weight_decay,
                        eps=oc.adamw_eps,
                    )
                ]
            case "microadam-new":
                # the version that optimizes only the linear layers with MicroAdam
                # and the other layers (embeds, layer norms) with AdamW
                # implemented after discussing with Frederik Kunstner
                return [
                    MicroAdam(
                        params=self._get_linear_layers(),
                        lr=oc.microadam_lr,
                        m=oc.microadam_m,
                        quant_block_size=oc.microadam_q_block,
                        k_init=oc.microadam_k,
                        alpha=0, # 0 means sparse update
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        weight_decay=oc.weight_decay,
                        eps=oc.adamw_eps,
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
                        lr=oc.adamw_lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps,
                    )
                ]
            case "microadam-torch":
                 opts = [
                    MicroAdamTorch(
                        params=self._get_linear_layers(),
                        lr=oc.microadam_lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps,
                        # *
                        num_grads=oc.microadam_m,
                        density=oc.microadam_k,
                        quant_info=oc.microadam_quant_info,
                        dtype_scales={"bf16": torch.bfloat16, "fp16": torch.float16}[oc.microadam_dtype_scales],
                        dtype_values={"bf16": torch.bfloat16, "fp16": torch.float16}[oc.microadam_dtype_values],
                        structured=oc.microadam_structured,
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
                        lr=oc.adamw_lr,
                        weight_decay=oc.weight_decay,
                        betas=(
                            oc.adamw_beta1,
                            oc.adamw_beta2,
                        ),
                        eps=oc.adamw_eps,
                    )
                ]
                 opts[0].set_param_name_mapping(mapping={p: name for name, p in self.model.named_parameters()})
                 return opts
            case "trion":
                return [
                    Trion(
                        params=self._get_linear_layers(),
                        lr=oc.trion_lr,
                        weight_decay=oc.weight_decay,
                        momentum=oc.trion_momentum,
                        # *
                        rank=oc.trion_rank,
                        scaling_type=oc.trion_scaling,
                        proj_type=oc.trion_proj,
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
                        lr=oc.adamw_lr,
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