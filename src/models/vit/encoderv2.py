import torch
import torch.nn as nn

from typing import Union
from functools import partial
from collections import OrderedDict

from models.vit.mlp import MLPBlock 
from models.common.sequential import VisionMultiInputSequential

class EncoderBlock(nn.Module):
    """ Generic Transformer encoder block.
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
        activation=nn.GELU
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation=activation)

        self.need_weights = need_weights # Whether to return attention weights as well

    def forward(self, input: torch.Tensor, key_padding_mask: Union[torch.Tensor, None]):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        result = None
        attention_weights = None # Needed only if self.need_weights is True for this specific Block

        # print(x0.shape)
        attended_input, attention_weights = self.self_attention(input, input, input, key_padding_mask=key_padding_mask, need_weights=self.need_weights)
        attended_input += input
        attended_input = self.ln_1(attended_input)
        attended_input_mlp = self.mlp(attended_input)
        attended_input_mlp += attended_input
        attended_input_ln = self.ln_2(attended_input_mlp)

        if self.need_weights:
            return attended_input_ln, attention_weights
        else: 
            return attended_input_ln


class Encoder(nn.Module):
    """ Transformer Model Encoder for sequence to sequence translation.
    """

    def __init__(
        self,
        seq_length: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        total_layers: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        need_weights: bool=False,
        activation=nn.GELU
    ):
        super().__init__()
        
        
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        # Need to load all the layers first to be able to load the weights
        layers = []
        for i in range(total_layers):


            layers.append(
                EncoderBlock(
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    norm_layer=norm_layer, 
                    need_weights=need_weights, 
                    activation=activation
                )
            )
        
            
        self.layers = VisionMultiInputSequential(*layers)


    def forward(self, input: torch.Tensor, key_padding_mask: Union[torch.Tensor, None]):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")

        input += self.pos_embedding
        
        # Take a subset of pretrained encoder layers
        result = self.layers(input, key_padding_mask)

        return result