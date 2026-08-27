from argparse import Namespace as ArgParseNamespace

from backends.base_backend import BaseBackend


class SingleBackend(BaseBackend):
    def __init__(self, cfg: ArgParseNamespace):
        super().__init__(
            cfg=cfg,
            rank=0,
            local_rank=0,
            world_size=1
        )
        if self.is_master:
            print(f"\n=== Using Single GPU backend ===\n")
