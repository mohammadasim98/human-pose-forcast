import torch
import torch.nn as nn

from typing import Union
from functools import partial

from models.vit.mlp import MLPBlock 
from models.common.sequential import FusionMultiInputSequential



class FusionEncoderBlock(nn.Module):
    """ Fuse image and pose sequence embeddings via Spatial
        Cross-Attention
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights = False,
        activation=nn.GELU
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation)

        self.need_weights = need_weights 
        


    def forward(self, memory_pose: torch.Tensor, memory_img: torch.Tensor, img_mask: Union[torch.Tensor, None]=None):
        """Perform forward pass

        Args:
            root_joints (torch.tensor): An input of shape (batch_size*sequence_length,  E) 
                or (batch_size, E)
            relative_poses (torch.tensor): An input of shape (batch_size*sequence_length, num_joints, E) 
                or (batch_size, num_joints, E)

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(memory_pose.dim() == 3, f"Expected (batch_size*sequence_length, num_joints+1, E) or (batch_size, num_joints+1, E) got {memory_pose.shape}")
        
        _, num_joints, hidden_dim = memory_pose.shape
        result = None
        attention_weights = None 

        attended_pose, attention_weights = self.self_attention(memory_pose, memory_img, memory_img, need_weights=self.need_weights, key_padding_mask=img_mask)
        attended_pose += memory_pose
        attended_pose = self.ln_1(attended_pose)
        attended_pose_mlp = self.mlp(attended_pose)
        attended_pose_mlp += attended_pose
        attended_pose_ln = self.ln_2(attended_pose_mlp)

        return attended_pose_ln, attention_weights
    
class FusionEncoder(nn.Module):
    """ Fusion Encoder
    """
    
    def __init__(
        self, 
        num_layers: int,
        num_heads: int, 
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights = False,
        activation=nn.GELU
        ) -> None:
        
        super().__init__()
        
        #######################################################
        # TODO: Need to implement a pose encoder
        # self.block = PoseEncoderBlock(...)
        layers = []
        for i in range(num_layers):
            layers.append(FusionEncoderBlock(num_heads, hidden_dim, mlp_dim, norm_layer, dropout, need_weights, activation=activation))
        
        self.layers = FusionMultiInputSequential(*layers)
        
    def forward(self, memory_pose: torch.Tensor, memory_img: torch.Tensor, img_mask: Union[torch.Tensor, None]=None):
        """Perform forward pass

        Args:
            root_joints (torch.tensor): An input of shape (batch_size*sequence_length,  E) 
                or (batch_size, E)
            relative_poses (torch.tensor): An input of shape (batch_size*sequence_length, num_joints, E) 
                or (batch_size, num_joints, E)

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(memory_pose.dim() == 3, f"Expected (batch_size*sequence_length, num_joints+1, E) or (batch_size, num_joints+1, E) got {memory_pose.shape}")
        torch._assert(memory_img.dim() == 3, f"Expected (batch_size*sequence_length, num_patches+1, E) or (batch_size, num_patches+1, E) got {memory_img.shape}")
        torch._assert(img_mask.dim() == 2, f"Expected (batch_size*sequence_length, num_patches+1) or (batch_size, num_patches+1) got {img_mask.shape}")

        
        
        result, attentions = self.layers(memory_pose, memory_img, img_mask)
        
        return result, attentions  


        