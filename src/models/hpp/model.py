import torch
import torchvision
import torch.nn as nn
from ..base_model import BaseModel
from ..vit import model as vit_model


class HPP_Predictor(BaseModel):

    def __int__(self, image_size, activation, num_layers, num_heads, hidden_dim,
                mlp_dim, patch_size=9, drop_prob=0.0):
        super.__init__()
        self.image_size = image_size
        self.activation = activation
        self.drop_prob = drop_prob
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim

    def build_model(self):
        self.root_spatial_vit = vit_model.VisionTransformer(self.image_size, self.patch_size,
                                                            self.num_layers, self.num_heads,
                                                            self.hidden_dim, self.mlp_dim)
        self.pos_spatial_vit = vit_model.VisionTransformer(self.image_size, self.patch_size,
                                                           self.num_layers, self.num_heads,
                                                           self.hidden_dim, self.mlp_dim)
        self.root_temp = vit_model.VisionTransformer(self.image_size, self.patch_size,
                                                     self.num_layers, self.num_heads,
                                                     self.hidden_dim, self.mlp_dim)
        self.pos_temp = vit_model.VisionTransformer(self.image_size, self.patch_size,
                                                           self.num_layers, self.num_heads,
                                                           self.hidden_dim, self.mlp_dim)
