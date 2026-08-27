from argparse import Namespace as ArgParseNamespace

from .base_backend import BaseBackend
from .ddp_backend import DDPBackend
from .fsdp2_backend import FSDP2Backend
from .single_backend import SingleBackend

def get_backend(cfg: ArgParseNamespace):
    match cfg.training.backend:
        case "single":
            return SingleBackend(cfg)
        case "ddp":
            return DDPBackend(cfg)
        case "fsdp2":
            return FSDP2Backend(cfg)
        case _:
            raise NotImplementedError(f"Backend {cfg.training.backend} is not implemented")
