import torch
import torchvision
import torch.nn as nn

from torchvision.ops.misc import MLP

from models.vit.mlp import  MLPBlock

class LinearProjectionBlock(nn.module):
    
    def __init__(
        self,
        mlps: list, 
        activation: str,
        inp_dim: int=2,
        out_dim: int=3
    ) -> None:
        
        self.mlps = mlps
        self.activations = activations
        
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
        in_dim = inp_dim
        for i, num in enumerate(mlps):
            layers.append(MLP(in_dim, [num], activation_layer=activation_layer))
            in_dim
        
        

class LinearProjection(nn.module):
    
    def __init__(
        self,
        mlps: list, 
        activation: str,
        inp_dim: int=2,
        out_dim: int=3,
        num_layers: int=4
    ) -> None:
        
        self.mlp = mlp
        self.activations = activations
        
        self.mlps = mlps
        self.activations = activations
        
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
        in_dim = inp_dim
        for i, num in enumerate(num_layers):
            layers.append(MLP(in_dim, [*mlps, out_dim], activation_layer=activation_layer))
            in_dim
        
        
    
        
    
