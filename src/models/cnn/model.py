import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.modules import padding

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()


        self._activation = getattr(nn, activation["type"])                         #        
        self._norm_layer = getattr(nn, norm_layer["type"])                                               #


        self._activation = getattr(nn, activation["type"])(**activation["args"])
        self._norm_layer = getattr(nn, norm_layer["type"])
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.drop_prob = drop_prob

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):


        layers = []
        in_channels = self.input_size

        in_feature = self.input_size
        for out_feature in self.hidden_layers[:5]:
            layers.append(nn.Conv2d(in_feature, out_feature, 3, padding=1))
            layers.append(self._norm_layer(out_feature))
            #layers.append(nn.BatchNorm2d(out_feature))
            layers.append(nn.MaxPool2d(2, 2))
            layers.append(self._activation)
            layers.append(nn.Dropout(self.drop_prob))
            in_feature = out_feature

        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_feature, self.num_classes))
        layers.append(nn.Softmax(dim=-1))

        self.layers = nn.Sequential(*layers)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):

        fig = plt.figure(figsize=(10, 5))
        params = self.layers.parameters()
        for param in params:
            # param = param.transpose(1, -1)
            # param = param.transpose(1, 2)
            for i in range(param.shape[0]):
                plt.subplot(8, 16, i+1)
                plt.axis("off")
                plt.imshow(self._normalize(param[i, ...].cpu().detach().numpy()))
            plt.subplots_adjust(hspace=0.2, wspace=0.2)
           

            plt.show()
            break
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):

        x = self.layers(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
