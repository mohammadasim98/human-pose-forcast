import torch
import torchvision
import torch.nn as nn

from torchvision.ops.misc import MLP
from models.projection.mlp import MLPBlock
        
        

class LinearProjection(nn.Module):
    
    def __init__(
        self,
        mlps: list, 
        activation: str="gelu",
        inp_dim: int=2,
        out_dim: int=3,
        num_layers: int=4,
        device: str="cuda:0"
    ) -> None:
        super().__init__()
        
        layers = []
        
        if activation == "relu":
            activation_layer = nn.ReLU
        elif activation == "gelu":
            activation_layer = nn.GELU
        elif activation == "elu":
            activation_layer = nn.ELU
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid
        else:
            activation_layer = None
            
        self.inp_proj = MLPBlock(inp_dim, [mlps[0]], activation_layer=activation_layer).to(device)
        in_dim = mlps[0]
        for i in range(num_layers):
            layers.append(MLPBlock(in_dim, [*mlps], activation_layer=activation_layer).to(device))
            in_dim = mlps[-1]
            
        self.out_proj = MLPBlock(in_dim, [out_dim], activation_layer=activation_layer).to(device)

        self.layers = layers
    
        
    
    def forward(self, inp):
        b, n, j, _ = inp.shape
        inp = inp.view(b, n, -1)
        inputs = self.inp_proj(inp)

        for layer in self.layers:
            out = layer(inputs)
            inputs = inputs + out
        out = self.out_proj(out)
        return out.view(b, n, -1, 3)