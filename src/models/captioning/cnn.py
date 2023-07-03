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
        #########################
        # TODO 
        # Create a sequential model (named `self.resnet`) with all the layers of resnet except the last fc layer.
        # Add a linear layer (named `self.linear`) to bring resnet features down to embed_size. Don't put the self.linear into the Sequential module.
        cnn = list(resnet.children())[:-1]
        for layer in cnn:
            layer.requires_grad = False
        self.resnet = nn.Sequential(*cnn)
        self.linear = nn.Linear(2048, embed_size)
        #########################
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        #########################
        # TODO 
        # Run your input images through the modules you created above (input -> Sequential -> final linear -> self.bn)
        # Make sure to freeze the weights of the resnet layers
        # finally return the normalized features
        x = self.resnet(images)
        x = nn.Flatten()(x)
        result = self.linear(x)
        result = self.bn(result)

        return result
        #########################