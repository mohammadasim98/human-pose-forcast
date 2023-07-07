import torch
import torch.nn as nn

from functools import partial

from models.vit.mlp import MLPBlock 
from models.transformer.encoder import EncoderBlock, Encoder

class MultiInputSequential(nn.Sequential):
    """ A custom nn.Sequential model for multiple inputs and outputs
    """
    def forward(self, *inputs):

        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

class GlobalTemporalEncoderBlock(nn.Module):
    """ Encode a temporal sequence of global feature with self-attention
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
        
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)

        self.need_weights = need_weights # Whether to return attention weights as well

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
        x1, attention_weights = self.self_attention(x0, x0, x0, need_weights=self.need_weights)
        x2 = inputs + x1
        x3 = self.ln_2(x2)
        x4 = self.mlp(x3)
        result = x2 + x4

        
        return result

class LocalTemporalEncoderBlock(nn.Module):
    """ Encode a temporal sequence of local features with cross-sequence-attention
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights = False,
        reduce: bool=False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.reduce = reduce
        
        self.ln_q = norm_layer(hidden_dim)
        self.ln_kv = norm_layer(hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.need_weights = need_weights
         
        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim) 
         
    def forward(self, inputs: torch.Tensor):
        """Perform forward pass

        Args:
            inputs (torch.Tensor): A (B, Hw, Nf, E) tensor with Hw as the history 
                window, Nf as the number of local features and E as the embedding/hidden
                feature dimension.

        Raises:
            result (torch.Tensor): A (B, Hw', Nf, E) tensor if reduce is False else a (B, Nf, E) tensor.
        """
        b, hw, nf, e = inputs.shape
        torch._assert(inputs.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {inputs.shape}")

        
        ###############################################################################
        # TODO: Need to implement a forward method for local temporal attention block
        ###############################################################################
        
        # Temporal attention on local features
        attended_values = []
        
        # Use oldest feature sequence from history as the first query
        query = inputs[:, 0, :, :]        
        
        for i in range(1, hw-1):
            
            # Update query
            key_value = inputs[:, i+1, :, :]

            # Layer norm 
            query_ln = self.ln_q(query)
            key_value_ln = self.ln_kv(key_value)
            
            # Forward temporal attention on two subsequent local features at different timestep                   
            attended_value, attention_weights = self.attention(query_ln, key_value_ln, key_value_ln)
            
            #############################################################################################
            # TODO: Think about different ways we could implement/propagate attended values in such a 
            # forward attention mechanism.
            # - I though maybe it makes sense to add a residual after getting the attended values with 
            #   with the original key/value and then use it as a query in the next timestep  
            #############################################################################################
            
            # Update query
            query = key_value + attended_value
            
            # Append attended queries
            attended_values.append(query.unsqueeze(1))
        
        if self.reduce:
            x3 = self.ln_2(query)
            x4 = self.mlp(x3)
            result = query + x4
              
        else:
            result = torch.cat(attended_values, axis=1)
            result = self.ln_2(result)
            
          
        return result

        
class TemporalEncoderBlock(nn.Module):
    """ Encode a temporal sequence of local and global features
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights = False,
        reduce: bool=False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.reduce = reduce

        self.need_weights = need_weights 
        
        self.local_encoder = LocalTemporalEncoderBlock(
            num_heads=num_heads, 
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim, 
            norm_layer=norm_layer,
            dropout=dropout, 
            need_weights=need_weights,
            reduce=self.reduce
        )
        
        self.global_encoder = GlobalTemporalEncoderBlock(
            num_heads=num_heads, 
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim, 
            norm_layer=norm_layer,
            dropout=dropout, 
            need_weights=need_weights
        )
        
    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor):
        """Perform forward pass

        Args:
            local_feat (torch.Tensor): A (B, Hw, Nf, E) tensor with Hw as the history 
                window, Nf as the number of local features and E as the embedding/hidden
                feature dimension.
                
            global_feat (torch.Tensor): A (B, Hw, E) tensor with Hw as the history 
                window and E as the embedding/hidden feature dimension.

        Raises:
            local_result (torch.Tensor): A (B, Nf, E) or (B, Hw', Nf, E) tensor if reduce is False.
            global_result (torch.Tensor) A (B, Hw, E) tensor.
        """
        torch._assert(local_feat.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {local_feat.shape}")
        torch._assert(global_feat.dim() == 3, f"Expected Global Features of shape \
            (batch_size, seq_length, hidden_dim) got {global_feat.shape}")
        
        ######################################################################
        # TODO: Need to implement a forward method for temporal encoder block
        ######################################################################
        
        
        local_result = self.local_encoder(local_feat)
        global_result = self.global_encoder(global_feat)
        

        return local_result, global_result
        
class TemporalEncoder(nn.Module):
    """ Temporal Encoder for local and global features
    """
    
    def __init__(
        self, 
        num_layers: int,
        num_heads: int, 
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights: bool=False
        ) -> None:
        
        super().__init__()
        
        self.hidden_dim = hidden_dim
        #######################################################
        # TODO: Need to implement a temporal encoder
        # self.block = TemporalEncoderBlock(...)
        # ...
        #######################################################
        layers = []
        
        for i in range(num_layers):
            layers.append(
                TemporalEncoderBlock(
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    norm_layer=norm_layer,
                    dropout=dropout,
                    need_weights=need_weights,
                    reduce=True if i == (num_layers - 1) else False
                )
            )
        
        self.layers = MultiInputSequential(*layers)
                
    def forward(self, local_feat: torch.Tensor, global_feat: torch.Tensor):
        """Perform forward pass

        Args:
            local_feat (torch.Tensor): A (B, Hw, Nf, E) tensor with Hw as the history 
                window, Nf as the number of local features and E as the embedding/hidden
                feature dimension.
                
            global_feat (torch.Tensor): A (B, Hw, E) tensor with Hw as the history 
                window and E as the embedding/hidden feature dimension.

        Raises:
            local_result (torch.Tensor): A (B, Nf, E) tensor.
            global_result (torch.Tensor) A (B, Hw, E) tensor.
        """
        torch._assert(local_feat.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {local_feat.shape}")
        torch._assert(global_feat.dim() == 3, f"Expected Global Features of shape \
            (batch_size, seq_length, hidden_dim) got {global_feat.shape}")
        ######################################################################
        # TODO: Need to implement a forward method for temporal encoder
        ######################################################################
        
        return self.layers(local_feat, global_feat)
        