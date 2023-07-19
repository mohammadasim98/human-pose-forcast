import torch
import torch.nn as nn

from functools import partial

from models.vit.mlp import MLPBlock 


class GlobalTemporalAttention(nn.Module):
    """ Encode a temporal sequence of global feature with self-attention
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights: bool=False,
        average_attn_weights: bool=True
    ) -> None:
        """Initialize Global Bi-directional Temporal Encoder

        Args:
            num_heads (int): Number of heads for the temporal attention module
            hidden_dim (int): Hidden dimension for the temporal attention module
            mlp_dim (int): MLP dimension for the temporal attention module
            norm_layer (torch.nn.Module, optional): Layer norm after temporal attention. 
                Defaults to partial(nn.LayerNorm, eps=1e-6).
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            need_weights (bool, optional): If true, return attention weights (not configured for now). 
                Defaults to False.
        """
        super().__init__()
        
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)

        self.need_weights = need_weights # Whether to return attention weights as well
        self.average_attn_weights = average_attn_weights
        
    def forward(self, inputs: torch.Tensor):
        """Perform forward pass

        Args:
            inputs (torch.Tensor): A (B, Hw, E) tensor with Hw as the history 
                window and E as the embedding/hidden feature dimension.

        Raises:
            result (torch.Tensor): A (B, Hw, E) tensor.
        """
        torch._assert(inputs.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {inputs.shape}")

        result = None
        attention_weights = None # Needed only if self.need_weights is True for this specific Block

        x0 = self.ln_1(inputs)
        x1, attention_weights = self.self_attention(
            x0, 
            x0, 
            x0, 
            need_weights=self.need_weights, 
            average_attn_weights=self.average_attn_weights
        )
        x2 = inputs + x1
        x3 = self.ln_2(x2)
        x4 = self.mlp(x3)
        result = x2 + x4

        
        return result, attention_weights
