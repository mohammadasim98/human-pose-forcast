import torch
import torch.nn as nn

from functools import partial


from models.vit.mlp import MLPBlock 
from models.transformer.encoder import EncoderBlock, Encoder

class PoseDecoderBlock(nn.Module):
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
        batch_first: bool=True,
        need_weights = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        # # Attention block
        # self.ln_1 = norm_layer(hidden_dim)
        # self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)

        # # MLP block
        # self.ln_2 = norm_layer(hidden_dim)
        # self.mlp = MLPBlock(hidden_dim, mlp_dim)

        # self.need_weights = need_weights 
        #######################################################
        # TODO: Need to implement a pose decoder
        # ...
        #######################################################
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        """Perform forward pass

        Args:
            inputs (torch.tensor): A (B, J, E) of embedding of each 2D pose keypoints.

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, {self.hidden_dim}) got {input.shape}")
        
        ######################################################################
        # TODO: Need to implement a forward method for pose decoder block
        ######################################################################
        raise NotImplementedError
    
    
class PoseDecoder(nn.Module):
    """ Pose Encoder
    """
    
    def __init__(
        self, 
        num_heads: int, 
        hidden_dim: int,
        dropout: float=0.0,
        batch_first: bool=True
        ) -> None:
        
        super().__init__()
        
        #######################################################
        # TODO: Need to implement a pose decoder block
        # self.block = PoseDecoderBlock(...)
        # ...
        #######################################################
        raise NotImplementedError
        
    def forward(self, inputs):
        """Perform forward pass

        Args:
            inputs (torch.tensor): A (B, J, E) of embedding of each 2D pose keypoints.

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(input.dim() == 3, f"Expected (batch_size, num_joints, hidden_dim) got {input.shape}")

        ######################################################################
        # TODO: Need to implement a forward method for pose decoder
        ######################################################################
        raise NotImplementedError
        