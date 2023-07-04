import torch
import torch.nn as nn

from functools import partial

from models.vit.mlp import MLPBlock 
from models.transformer.encoder import EncoderBlock, Encoder




class TemporalEncoderBlock(nn.Module):
    """ Encode a temporal sequence embeddings via Temporal
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

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)

        self.need_weights = need_weights 
        
        #######################################################
        # TODO: Need to implement a temporal encoder block
        # ...
        #######################################################
        raise NotImplementedError

    def forward(self, input: torch.Tensor):
        """Perform forward pass

        Args:
            inputs (torch.tensor): A (B, N, LE x GE) Concatenated Local and Global embedding LE and GE resp.
                as a single embedding

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        ######################################################################
        # TODO: Need to implement a forward method for temporal encoder block
        ######################################################################
        raise NotImplementedError
        
class TemporalEncoder(nn.Module):
    """ Temporal Encoder
    """
    
    def __init__(
        self, 
        num_heads: int, 
        hidden_dim: int,
        dropout: float=0.0,
        batch_first: bool=True
        ) -> None:
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        #######################################################
        # TODO: Need to implement a temporal encoder
        # self.block = TemporalEncoderBlock(...)
        # ...
        #######################################################
        raise NotImplementedError
        
    def forward(self, inputs):
        """Perform forward pass

        Args:
            inputs (torch.tensor): A (B, N, LE x GE) Concatenated Local and Global embedding LE and GE resp.
                as a single embedding

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, {self.hidden_dim}) got {input.shape}")

        ######################################################################
        # TODO: Need to implement a forward method for temporal encoder
        ######################################################################
        raise NotImplementedError
        