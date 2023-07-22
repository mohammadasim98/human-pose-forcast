import torch
import torch.nn as nn

from typing import Union
from functools import partial

from models.vit.mlp import MLPBlock 
from models.common.sequential import PoseMultiInputSequential, MultiInputSequential
# class PoseEncoderBlock(nn.Module):
#     """ Encode a pose keypoint sequence embeddings via Spatial
#         Self-Attention
#     """

#     def __init__(
#         self,
#         num_heads: int,
#         hidden_dim: int,
#         mlp_dim: int,
#         norm_layer = partial(nn.LayerNorm, eps=1e-6),
#         dropout: float=0.0,
#         need_weights = False,
#     ):
#         super().__init__()
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
        
#         # Attention block
#         self.ln_1 = norm_layer(hidden_dim)
#         self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

#         # MLP block
#         self.ln_2 = norm_layer(hidden_dim)
#         self.mlp = MLPBlock(hidden_dim, mlp_dim)

#         self.need_weights = need_weights 
        
#     def _generate_key_padding_mask(self, relative_poses: torch.Tensor) -> torch.Tensor:
#         raise NotImplementedError

#     def forward(self, root_joints: torch.Tensor, relative_poses: torch.Tensor, key_padding_mask: Union[torch.Tensor, None]=None):
#         """Perform forward pass

#         Args:
#             root_joints (torch.tensor): An input of shape (batch_size*sequence_length,  E) 
#                 or (batch_size, E)
#             relative_poses (torch.tensor): An input of shape (batch_size*sequence_length, num_joints, E) 
#                 or (batch_size, num_joints, E)

#         Raises:
#             NotImplementedError: Need to implement forward pass
#         """
#         torch._assert(relative_poses.dim() == 3, f"Expected (batch_size*sequence_length, num_joints, E) or (batch_size, num_joints, E) got {relative_poses.shape}")
#         torch._assert(root_joints.dim() == 2, f"Expected (batch_size*sequence_length, num_joints, E) or (batch_size, num_joints, E) got {root_joints.shape}")
        
#         _, num_joints, hidden_dim = relative_poses.shape

#         result = None
#         attention_weights = None 
#         input = torch.cat([root_joints.unsqueeze(1), relative_poses], dim=1)
#         x0 = self.ln_1(input)
#         x1, attention_weights = self.self_attention(x0, x0, x0, key_padding_mask=key_padding_mask, need_weights=self.need_weights)
#         x2 = input + x1
#         x3 = self.ln_2(x2)
#         x4 = self.mlp(x3)
#         result = x2 + x4

#         return result


class PoseEncoderBlock(nn.Module):
    """ Encode a pose keypoint sequence embeddings via Spatial
        Self-Attention
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)

        self.need_weights = need_weights 
        


    def forward(self, pose: torch.Tensor, pose_mask: Union[torch.Tensor, None]=None):
        """Perform forward pass

        Args:
            root_joints (torch.tensor): An input of shape (batch_size*sequence_length,  E) 
                or (batch_size, E)
            relative_poses (torch.tensor): An input of shape (batch_size*sequence_length, num_joints, E) 
                or (batch_size, num_joints, E)

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(pose.dim() == 3, f"Expected (batch_size*sequence_length, num_joints+1, E) or (batch_size, num_joints+1, E) got {pose.shape}")
        
        _, num_joints, hidden_dim = pose.shape
        result = None
        attention_weights = None 
        x0 = self.ln_1(pose)
        x1, attention_weights = self.self_attention(x0, x0, x0, need_weights=self.need_weights, key_padding_mask=pose_mask)
        x2 = pose + x1
        x3 = self.ln_2(x2)
        x4 = self.mlp(x3)
        result = x2 + x4

        return result
    
class PoseEncoder(nn.Module):
    """ Pose Encoder
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
        ) -> None:
        
        super().__init__()
        
        #######################################################
        # TODO: Need to implement a pose encoder
        # self.block = PoseEncoderBlock(...)
        layers = []
        for i in range(num_layers):
            layers.append(PoseEncoderBlock(num_heads, hidden_dim, mlp_dim, norm_layer, dropout, need_weights))
        
        self.layers = PoseMultiInputSequential(*layers)
        
    def forward(self, root_joints: torch.Tensor, relative_poses: torch.Tensor, pose_mask: Union[torch.Tensor, None]=None):
        """Perform forward pass

        Args:
            root_joints (torch.tensor): An input of shape (batch_size*sequence_length,  E) 
                or (batch_size, E)
            relative_poses (torch.tensor): An input of shape (batch_size*sequence_length, num_joints, E) 
                or (batch_size, num_joints, E)

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(relative_poses.dim() == 3, f"Expected (batch_size*sequence_length, num_joints, E) or (batch_size, num_joints, E) got {relative_poses.shape}")
        torch._assert(root_joints.dim() == 2, f"Expected (batch_size*sequence_length, num_joints, E) or (batch_size, num_joints, E) got {root_joints.shape}")        
        
        
        input = torch.cat([root_joints.unsqueeze(1), relative_poses], dim=1)

        result = self.layers(input, pose_mask)
        
        return result[:, 1:, ...], result[:, 0, ...]

        