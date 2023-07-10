import torchvision
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(self, layer_config, num_classes, activation, norm_layer, fine_tune, weigths='DEFAULT'):
        super(VGG11_bn, self).__init__()



        self._activation = getattr(nn, activation["type"])(**activation["args"])
        self._norm_layer = getattr(nn, norm_layer["type"])
        self.num_classes = num_classes
        self.layer_config = layer_config
        self.fine_tune = fine_tune

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._weights = weigths
        self._build_model()

    def _build_model(self):

        self.vgg11_bn = torchvision.models.vgg11_bn(weights=self._weights)
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        vgg_conv = self.vgg11_bn.features
        if not self.fine_tune:
            for layer in vgg_conv:
                layer.requires_grad = False

        layers = [nn.Flatten()]
        in_features = self.layer_config[0]
        for layer in self.layer_config:
            layers.append(nn.Linear(in_features, layer))
            layers.append(self._norm_layer(layer))
            layers.append(self._activation)
            in_features = layer

        layers.append(nn.Softmax(dim=-1))

        self.layers = nn.Sequential(vgg_conv, *layers)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):


        x = self.layers(x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x