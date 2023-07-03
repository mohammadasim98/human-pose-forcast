import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.modules import padding

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ######################################################################################################
        # TODO: Initialize the different model parameters from the config file                               #    
        # You can use the arguments given in the constructor. For activation and norm_layer                  #
        # to make it easier, you can use the following two lines                                             #                              
        self._activation = getattr(nn, activation["type"])                         #        
        self._norm_layer = getattr(nn, norm_layer["type"])                                               #
        # Or you can just hard-code using nn.Batchnorm2d and nn.ReLU as they remain fixed for this exercise. #
        ###################################################################################################### 
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self._activation = getattr(nn, activation["type"])(**activation["args"])
        self._norm_layer = getattr(nn, norm_layer["type"])
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.drop_prob = drop_prob

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module.                          #
        #################################################################################
        layers = []
        in_channels = self.input_size
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
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
        #################################################################################
        # TODO: Implement the forward pass computations                                 #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        x = self.layers(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x
