import torch.nn as nn
from ..base_model import BaseModel


class MultiLayerPerceptron(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, drop_prob=0.0):
        super(MultiLayerPerceptron, self).__init__()
        

        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.drop_prob = drop_prob
        self.build_model()
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def build_model(self):



        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        in_feature = self.input_size
        layers.append(nn.Flatten(1, -1))
        for out_feature in self.hidden_layers:
            layers.append(nn.Linear(in_feature, out_feature))
            layers.append(nn.ReLU())
            in_feature = out_feature
        
        layers.append(nn.Linear(in_feature, self.num_classes))
        layers.append(nn.Softmax(dim=-1))
        
        self.layers = nn.Sequential(*layers)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    def forward(self, x):

        
        x = self.layers(x)
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return x