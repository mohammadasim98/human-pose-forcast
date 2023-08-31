import torch
import torch.nn as nn

from typing import Union
from functools import partial

from models.common.sequential import TemporalMultiInputSequentialV3
from models.temporal.temporal_attention import LocalBackwardTemporalAttention, LocalForwardTemporalAttention, BidirectionalTemporalAttention

      
class TemporalEncoderBlock(nn.Module):
    """ Encode a temporal sequence of local and global features
    """

    def __init__(
        self,
        direction: str, # "forward", "backward", or "both" (not configured yet)
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        reduce: bool=False,
        dropout: float=0.0,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        need_weights: bool=False,
        average_attn_weights: bool=True,
        activation=nn.GELU,
        num_lstm_layers: int=3,
        use_lstm: bool=False,
        num_layers: int=3,
        num_query: int=18
    ):
        """Initialize Temporal Encoder Block

        Args:
            num_heads (int): Number of heads for the temporal attention module
            hidden_dim (int): Hidden dimension for the temporal attention module
            mlp_dim (int): MLP dimension for the temporal attention module
            norm_layer (torch.nn.Module, optional): Layer norm after temporal attention. 
                Defaults to partial(nn.LayerNorm, eps=1e-6).
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            need_weights (bool, optional): If true, return attention weights (not configured for now). 
                Defaults to False.
            reduce (bool, optional): If true, return the propagated attended values. Else return all the attented values. 
                Defaults to False.  
            directions (list): A string of direction "forward" or "backward" 
                for the temporal attention module
        """
        torch._assert(direction in ["forward", "backward", "bidirectional"], "Invalid direction value")
        
        super().__init__()
        self.num_heads = num_heads
        self.reduce = reduce
        self.direction = direction
        self.need_weights = need_weights 
        
        if self.direction == "forward":
            self.local_forward_attention = LocalForwardTemporalAttention(
                num_heads=num_heads, 
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim, 
                norm_layer=norm_layer,
                reduce=reduce,
                dropout=dropout, 
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                activation=activation,
                use_lstm=use_lstm,
                num_layers=num_layers,
                num_lstm_layers=num_lstm_layers,
                num_query=num_query
            )
        if self.direction == "backward":

            self.local_backward_attention = LocalBackwardTemporalAttention(
                num_heads=num_heads, 
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim, 
                norm_layer=norm_layer,
                reduce=reduce,
                dropout=dropout, 
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                activation=activation,
                use_lstm=use_lstm,
                num_layers=num_layers,
                num_lstm_layers=num_lstm_layers,
                num_query=num_query

            )

        if self.direction == "bidirectional":
            self.bidirectional_attention = BidirectionalTemporalAttention(
                num_heads=num_heads, 
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim, 
                norm_layer=norm_layer,
                reduce=reduce,
                dropout=dropout, 
                need_weights=need_weights,
                average_attn_weights=average_attn_weights,
                activation=activation,
                use_lstm=use_lstm,
                num_layers=num_layers,
                num_lstm_layers=num_lstm_layers,
                num_query=num_query

                )
        print("Using LSTM: ", use_lstm)

                
    def forward(self, feat: torch.Tensor, mask: Union[torch.Tensor, None]=None, states: Union[tuple, None]=None):
        """Perform forward pass

        Args:
            local_feat (torch.Tensor): A (B, Hw, Nf, E) tensor with Hw as the history 
                window, Nf as the number of local features and E as the embedding/hidden
                feature dimension.
                
            global_feat (torch.Tensor): A (B, Hw, E) tensor with Hw as the history. 
                window and E as the embedding/hidden feature dimension.

        Returns:
            local_result (torch.Tensor): A (B, Nf, E) or (B, Hw', Nf, E) tensor if reduce is False.
            global_result (torch.Tensor) A (B, Hw, E) tensor.
        """
        torch._assert(feat.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {feat.shape}")
        
        
        ##########################################################################
        # TODO: Need to think about forward-backward or backward-forward methods
        
        if self.direction == "forward":
            local_result, states, attention_weights  = self.local_forward_attention(feat, mask, states=states)
            
        elif self.direction == "backward":
            local_result, states, attention_weights = self.local_backward_attention(feat, mask, states=states)
            
        elif self.direction == "bidirectional":
            local_result, states, attention_weights  = self.bidirectional_attention(feat, mask, forward_states=states, backward_states=states)


        return local_result, states, attention_weights
        
class TemporalEncoderV3(nn.Module):
    """ Temporal Encoder for local and global features
    """
    
    def __init__(
        self, 
        num_heads: int, 
        hidden_dim: int,
        mlp_dim: int,
        directions: list,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights: bool=False,
        average_attn_weights: bool=True,
        activation=nn.GELU,
        num_lstm_layers: int=3,
        use_lstm: bool=False,
        num_layers: int=3,
        num_query: int=18
        ) -> None:
        """Initialize Temporal Encoder

        Args:
            num_layers (int): Number of TemporalEncoderBlock
            num_heads (int): Number of heads for all TemporalEncoderBlock
            hidden_dim (int): Hidden dimension for all TemporalEncoderBlock
            mlp_dim (int): MLP dimension for all TemporalEncoderBlock
            directions (list): A list strings of direction "forward" or "backward" 
                for the temporal attention for each TemporalEncoderBlock
            norm_layer (torch.nn.Module, optional): Layer norm after attention. 
                Defaults to partial(nn.LayerNorm, eps=1e-6).
            dropout (float, optional): Dropout probability. Defaults to 0.0.
            need_weights (bool, optional): If true, return attention weights (not configured for now). 
                Defaults to False.
        """
        super().__init__()
        
        #######################################################
        # TODO: Need to implement a temporal encoder

        layers = []
        
        for i, direction in enumerate(directions):
            layers.append(
                TemporalEncoderBlock(
                    direction=direction,
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    reduce=True if i == (len(directions) - 1) else False,
                    dropout=dropout,
                    norm_layer=norm_layer,
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    activation=activation,
                    use_lstm=use_lstm,
                    num_layers=num_layers,
                    num_lstm_layers=num_lstm_layers,
                    num_query=num_query
                )
            )
        
        self.layers = TemporalMultiInputSequentialV3(*layers)
                
    def forward(self, feat: torch.Tensor, mask: Union[torch.Tensor, None]=None, states: Union[tuple, None]=None):
        """Perform forward pass

        Args:
            local_feat (torch.Tensor): A (B, Hw, Nf, E) tensor with Hw as the history 
                window, Nf as the number of local features and E as the embedding/hidden
                feature dimension.
                
            global_feat (torch.Tensor): A (B, Hw, E) tensor with Hw as the history 
                window and E as the embedding/hidden feature dimension.

        Returns:
            local_result (torch.Tensor): A (B, Nf, E) tensor.
            global_result (torch.Tensor) A (B, Hw, E) tensor.
        """
        torch._assert(feat.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {feat.shape}")


        return self.layers(feat, mask=mask, states=states)

        