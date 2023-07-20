
import torch.nn as nn
from torchvision.ops.misc import MLP

class MLPBlock(MLP):
    """ Generic Transformer MLP block.
    """

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: list, activation_layer=nn.GELU):
        super().__init__(in_dim, mlp_dim, activation_layer=activation_layer, inplace=None, dropout=0.0)

    