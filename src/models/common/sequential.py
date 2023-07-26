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

class VisionMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, inputs, key_padding_mask):
        for module in self._modules.values():
            inputs = module(inputs, key_padding_mask)

        return inputs    
    
class PoseMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, inputs, key_padding_mask):
        for module in self._modules.values():
            inputs = module(inputs, key_padding_mask)

        return inputs
    
class TemporalMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, *inputs, mask):
        attentions = []
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs[:2], mask=mask)
                attentions.append(inputs[-1])
                inputs = inputs[:2]
            else:
                inputs = module(inputs, mask=mask)
                attentions.append(inputs[-1])

            
        return *inputs, attentions