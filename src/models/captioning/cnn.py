import torch
import torchvision
import torch.nn as nn


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """
        Load a pretrained ResNet-152 and modify top layers to extract features
        """
        super(EncoderCNN, self).__init__()
        
        resnet = torchvision.models.resnet152(pretrained=True)

        cnn = list(resnet.children())[:-1]
        for layer in cnn:
            layer.requires_grad = False
        self.resnet = nn.Sequential(*cnn)
        self.linear = nn.Linear(2048, embed_size)
        #########################
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""

        x = self.resnet(images)
        x = nn.Flatten()(x)
        result = self.linear(x)
        result = self.bn(result)

        return result
        #########################