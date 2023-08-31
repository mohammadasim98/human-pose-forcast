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
    
    
class TemporalMultiInputSequentialV2(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, inputs, mask):
        attentions = []
        for module in self._modules.values():
            inputs, attention = module(feat=inputs, mask=mask)
            attentions.append(attention)

        return inputs, attentions
    
class TemporalMultiInputSequentialV3(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, inputs, mask, states):
        attentions = []
        for module in self._modules.values():
            inputs, states, attention = module(feat=inputs, mask=mask, states=states)
            attentions.append(attention)

        return inputs, states, attentions
    
    
    
class AttentionMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, query, key, value, mask):
        attentions = []
        for i, module in enumerate(self._modules.values()):
            query, attention = module(query, key, value, mask)

            if attention is not None:   
                attentions.append(attention)

            
        return query, attentions
    
    
class FusionMultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, memory_pose, memory_img, img_mask):
        attentions = []
        for i, module in enumerate(self._modules.values()):
            memory_pose, attention = module(memory_pose, memory_img, img_mask)

            if attention is not None:   
                attentions.append(attention)

            
        return memory_pose, attentions