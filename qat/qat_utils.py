import torch
import torch.nn as nn
from argparse import Namespace as ArgParseNamespace
from .quartet_v2 import Quartet_II_Linear


def swap_module(model, module_name, new_module):
    name_parts = module_name.split('.')
    parent = model
    for part in name_parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last_part = name_parts[-1]
    if last_part.isdigit():
        parent[int(last_part)] = new_module
    else:
        setattr(parent, last_part, new_module)

def is_wrapped(model):
    wrapper_modules_tuple = (
        # BackWushVanillaLinear,
        # BackWushIdentityLinear,
        # BackWushHadamardLinear,
        # BackWushShmpLinear,
        Quartet_II_Linear,
    )
    for _, module in model.named_modules():
        if isinstance(module, wrapper_modules_tuple):
            return True
    return False

def wrap_model_for_qat(model, cfg: ArgParseNamespace, except_modules=None):
    if cfg.qat.qat_method == "disabled":
        return

    if except_modules is None:
        except_modules = [
            "token_embed",
            "lm_head",
        ]

    assert cfg is not None, "config is required"
    assert not is_wrapped(model), "model is already wrapped"

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any([exc in name for exc in except_modules]):
                print(f"skipped wrapping {name}")
                continue

            if cfg.qat.qat_method == 'quartet2':
                wrapped_module = Quartet_II_Linear.from_unquantized(module, name=name, hadamard_dim=cfg.qat.hadamard_size)
            else:
                raise RuntimeError(f"Unknown qat.qat_method value: {cfg.qat.qat_method}")
            # setattr(wrapped_module, 'layer_name', name)
            swap_module(model, name, wrapped_module)
            # print(f"wrapped {name}")

