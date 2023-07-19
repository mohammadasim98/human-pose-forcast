import torch
import torchvision
import torch.nn as nn

from torchvision.ops.misc import MLP

        
        

class LinearProjection(nn.Module):
    
    def __init__(
        self,
        mlps: list, 
        activation: str="gelu",
        inp_dim: int=2,
        out_dim: int=3,
        num_layers: int=4
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
            
        self.inp_proj = MLP(inp_dim, [mlps[0]], activation_layer=activation_layer)
        in_dim = mlps[0]
        for i in range(num_layers):
            layers.append(MLP(in_dim, [*mlps], activation_layer=activation_layer))
            in_dim = mlps[-1]
            
        self.out_proj = MLP(in_dim, [out_dim], activation_layer=activation_layer)

        self.layers = layers
    
        
    
    def forward(self, inp):
        
        inputs = self.inp_proj(inp)
        
        for layer in self.layers:
            out = layer(inputs)
            inputs = inputs + out
        
        return self.out_proj(out)