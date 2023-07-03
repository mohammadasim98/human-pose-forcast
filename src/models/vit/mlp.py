
import torch.nn as nn
from torchvision.ops.misc import MLP

class MLPBlock(MLP):
    """ Generic Transformer MLP block.
    """

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=0.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        for i in range(2):
            for type in ["weight", "bias"]:
                old_key = f"{prefix}linear_{i+1}.{type}"
                new_key = f"{prefix}{3*i}.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )