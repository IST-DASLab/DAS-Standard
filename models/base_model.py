import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import Namespace as ArgParseNamespace

class BaseModel(nn.Module):
    def __init__(self, cfg: ArgParseNamespace):
        super(BaseModel, self).__init__()
        self.cfg = cfg