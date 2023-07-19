import torch.nn as nn




class MultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, *inputs):

        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
class PoseMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, pose, key_padding_mask):
        for module in self._modules.values():
            inputs = module(pose, key_padding_mask)

        return inputs, key_padding_mask
    
class TemporalMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, *inputs):
        attentions = []
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs[:2])
                attentions.append(inputs[-1])
                inputs = inputs[:2]
            else:
                inputs = module(inputs)
                attentions.append(inputs[-1])

            
        return *inputs, attentions