import torch
import torch.nn as nn

from functools import partial

from models.vit.mlp import MLPBlock 


class LocalForwardTemporalAttention(nn.Module):
    """ Encode a temporal sequence of local features with cross-sequence-attention
        in forward direction
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
    ) -> None:
        """Initialize Local Forward Temporal Encoder

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
        """
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
        torch._assert(inputs.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {inputs.shape}")

        
        ######################################################################################
        # TODO: Need to implement a forward method for local forward temporal attention block
        
        b, hw, nf, e = inputs.shape 
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
            if len(attended_values):    
                result = torch.cat(attended_values, axis=1)
                result = self.ln_2(result)

            else: 
                result = query.unsqueeze(1)
            
          
        return result


class LocalBackwardTemporalAttention(nn.Module):
    """ Encode a temporal sequence of local features with cross-sequence-attention
        in backward direction
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
    ) -> None:
        """Initialize Local Backward Temporal Encoder

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
        """
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

        Returns:
            result (torch.Tensor): A (B, Hw', Nf, E) tensor if reduce is False else a (B, Nf, E) tensor.
        """
        torch._assert(inputs.dim() == 4, f"Expected Local Features of shape \
            (batch_size, seq_length, num_feature, hidden_dim) got {inputs.shape}")

        
        #######################################################################################
        # TODO: Need to implement a forward method for local backward temporal attention block
        b, hw, nf, e = inputs.shape

        # Temporal attention on local features
        attended_values = []
        
        # Use oldest feature sequence from history as the first query
        query = inputs[:, -1, :, :]        

        for i in range(hw-1, 1, -1):
            # Update key and value
            key_value = inputs[:, i-1, :, :]

            # Layer norm 
            query_ln = self.ln_q(query)
            key_value_ln = self.ln_kv(key_value)
            
            # Backward temporal attention on two subsequent local features at different timestep                   
            attended_value, attention_weights = self.attention(query_ln, key_value_ln, key_value_ln)
            
            #############################################################################################
            # TODO: Think about different ways we could implement/propagate attended values in such a 
            # Backward attention mechanism.
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
            if len(attended_values):    
                result = torch.cat(attended_values, axis=1)
                result = self.ln_2(result)

            else: 
                result = query.unsqueeze(1)
                
            
          
        return result