import torch.nn as nn




class MultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, inputs, key_padding_mask):
        for module in self._modules.values():
            inputs = module(inputs, key_padding_mask)

        return inputs