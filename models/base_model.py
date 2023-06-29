import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()

        ret_str += f"\n(layers): (parameters)"
        total = 0
        for i, param in enumerate(self.parameters()):
            if param.requires_grad:
                ret_str += f"\n({i}): {str(param.numel())}"
                total += param.numel()
        ret_str += f"\n(total): ({total})"
        return ret_str

