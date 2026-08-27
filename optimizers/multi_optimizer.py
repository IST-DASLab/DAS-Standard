import torch
import torch.nn as nn
from torch.optim import AdamW
from argparse import Namespace as ArgParseNamespace

from backends import BaseBackend
from lr_schedulers import get_lr_cos_warmup

from ista_daslab_optimizers import MicroAdam

# from .muon import MuonSingle, MuonDDP, MuonFSDP2


class MultiOptimizer:
    def __init__(self, cfg: ArgParseNamespace, model: nn.Module, backend: BaseBackend):
        self.cfg = cfg
        self.model = model
        self.backend = backend

        self.optim_iter = 0
        self.optimizers = self._create_optimizers_list()
        self._set_lr_init()

        self.zero_grad(set_to_none=True)

    def _create_optimizers_list(self):
        cfg = self.cfg

        match cfg.optimizer.optimizer_name:
            case "adamw":
                return [
                    AdamW(
                        params=self._get_all_params(),
                        lr=cfg.optimizer.adamw_lr,
                        weight_decay=cfg.optimizer.weight_decay,
                        betas=(
                            cfg.optimizer.adamw_beta1,
                            cfg.optimizer.adamw_beta2,
                        ),
                        eps=cfg.optimizer.adamw_eps),
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
                        lr=cfg.optimizer.muon_lr,
                        weight_decay=cfg.optimizer.weight_decay,
                        momentum=cfg.optimizer.muon_momentum
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
                        lr=cfg.optimizer.adamw_lr,
                        weight_decay=cfg.optimizer.weight_decay,
                        betas=(
                            cfg.optimizer.adamw_beta1,
                            cfg.optimizer.adamw_beta2,
                        ),
                        eps=cfg.optimizer.adamw_eps,
                    )
                ]
            case "microadam-paper":
                # the version that optimizes all layers wih MicroAdam
                return [
                    MicroAdam(
                        params=self.model.parameters(),
                        lr=cfg.optimizer.adamw_lr,
                        m=cfg.optimizer.microadam_m,
                        quant_block_size=cfg.optimizer.microadam_q_block,
                        k_init=cfg.optimizer.microadam_k,
                        alpha=0, # 0 means sparse update
                        betas=(
                            cfg.optimizer.adamw_beta1,
                            cfg.optimizer.adamw_beta2,
                        ),
                        weight_decay=cfg.optimizer.weight_decay,
                        eps=cfg.optimizer.adamw_eps,
                    )
                ]
            case "microadam-new":
                # the version that optimizes only the linear layers with MicroAdam
                # and the other layers (embeds, layer norms) with AdamW
                # implemented after discussing with Frederik Kunstner
                return [
                    MicroAdam(
                        params=self._get_linear_layers(),
                        lr=cfg.optimizer.adamw_lr,
                        m=cfg.optimizer.microadam_m,
                        quant_block_size=cfg.optimizer.microadam_q_block,
                        k_init=cfg.optimizer.microadam_k,
                        alpha=0, # 0 means sparse update
                        betas=(
                            cfg.optimizer.adamw_beta1,
                            cfg.optimizer.adamw_beta2,
                        ),
                        weight_decay=cfg.optimizer.weight_decay,
                        eps=cfg.optimizer.adamw_eps,
                    ),
                    AdamW(
                        params=self._get_input_embedding() + self._get_pos_embedding() + self._get_output_embedding() + self._get_layer_norms(),
                        lr=cfg.optimizer.adamw_lr,
                        weight_decay=cfg.optimizer.weight_decay,
                        betas=(
                            cfg.optimizer.adamw_beta1,
                            cfg.optimizer.adamw_beta2,
                        ),
                        eps=cfg.optimizer.adamw_eps,
                    )
                ]
            case _:
                raise NotImplementedError(f'Optimizer {cfg.optimizer.optimizer_name} is not implemented!')

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
                match cfg.optimizer.lr_sched_name:
                    case "cosine":
                        group["lr"] = get_lr_cos_warmup(
                            cfg=cfg,
                            lr_init=group["lr_init"],
                            step=self.optim_iter,
                        )
                    case _:
                        raise NotImplementedError(f"Scheduler {cfg.optimizer.lr_sched_name} is not implemented")

    def extract_learning_rates(self):
        lrs = {}
        for opt in self.optimizers:
            opt_name = opt.__class__.__name__
            lrs[f"lrs/{opt_name}"] = opt.param_groups[0]["lr"]
        return lrs

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