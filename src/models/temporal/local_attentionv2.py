import torch
import torch.nn as nn

from functools import partial
from typing import Union
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
        reduce: bool=False,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights: bool=False,
        average_attn_weights: bool=True,
        activation=nn.GELU,
        num_layers: int=3,
        use_lstm: bool=False
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
        self.average_attn_weights = average_attn_weights

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        
        self.mlpk = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        self.mlpq = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.res_ln = norm_layer(hidden_dim)
        self.use_lstm = use_lstm

        
         
    def forward(self, inputs: torch.Tensor, mask: Union[torch.Tensor, None]=None, states: Union[tuple, None]=None):
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
        attention_weights = []
        # Use oldest feature sequence from history as the first query
        query = inputs[:, 0, :, :]  
        # attended_values.append(query.unsqueeze(1))
        att_weights = None
        query_ln = self.ln_q(query)            
        for i in range(1, hw):
            # Update query
            key_value = inputs[:, i, :, :]
            state = key_value
            # Layer norm 
            key_value_ln = self.ln_kv(key_value)
            
            # query_ln = self.mlpq(query_ln)

            # Forward temporal attention on two subsequent local features at different timestep                   
            attended_value, att_weights = self.attention(
                query_ln, 
                key_value_ln, 
                key_value_ln, 
                need_weights=self.need_weights, 
                average_attn_weights=self.average_attn_weights,
                key_padding_mask=mask[:, i, :] if mask is not None and i < mask.shape[1] else None
            )
            # attention_weights shape: (1, N, N) where N is temporal sequence length
            #############################################################################################
            # TODO: Think about different ways we could implement/propagate attended values in such a 
            # forward attention mechanism.
            # - I though maybe it makes sense to add a residual after getting the attended values with 
            #   with the original key/value and then use it as a query in the next timestep  
            #############################################################################################
            if self.use_lstm:
                attended_value, states = self.lstm(attended_value, states)
            
            query = self.res_ln(self.mlpq(attended_value) + key_value_ln)
            
            # Append attended queries
            attended_values.append(query.unsqueeze(1))
            if att_weights is not None:
                attention_weights.append(att_weights)

        if len(attention_weights):
            attention_weights = torch.cat(attention_weights, dim=0)
        else:
            attention_weights = None
        
        if self.reduce:
            x3 = self.ln_2(query)
            x4 = self.mlp(x3)
            result = x4 + query
             
        else:
            if len(attended_values):    
                result = torch.cat(attended_values, axis=1)
                result = self.ln_2(result)
                result = self.mlp(result)

            else: 
                result = query.unsqueeze(1)
                
        
        return result, states, attention_weights


class LocalBackwardTemporalAttention(nn.Module):
    """ Encode a temporal sequence of local features with cross-sequence-attention
        in backward direction
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        reduce: bool=False,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights: bool=False,
        average_attn_weights: bool=True,
        activation=nn.GELU,
        num_layers: int=1,
        use_lstm: bool=False
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
        self.average_attn_weights = average_attn_weights
        
        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        self.mlpq = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
        self.use_lstm = use_lstm
        self.res_ln = norm_layer(hidden_dim)
        
    def forward(self, inputs: torch.Tensor, mask: Union[torch.Tensor, None]=None, states: Union[tuple, None]=None):
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
        attention_weights = []
        # Use oldest feature sequence from history as the first query
        query = inputs[:, -1, :, :]        
        # attended_values.append(query.unsqueeze(1))
        query_ln = self.ln_q(query)
        att_weights = None
        for i in range(hw-2, -1, -1):
            # Update key and value
            key_value = inputs[:, i, :, :]

            # Layer norm 
            key_value_ln = self.ln_kv(key_value)
            
            # query_ln = self.mlpq(query_ln)

            # Backward temporal attention on two subsequent local features at different timestep                   
            attended_value, att_weights = self.attention(
                query_ln, 
                key_value_ln, 
                key_value_ln, 
                need_weights=self.need_weights, 
                average_attn_weights=self.average_attn_weights,
                key_padding_mask=mask[:, i, :] if mask is not None and i < mask.shape[1] else None
            )
            
            # attention_weights shape: (1, N, N) where N is temporal sequence length
            #############################################################################################
            # TODO: Think about different ways we could implement/propagate attended values in such a 
            # Backward attention mechanism.
            # - I though maybe it makes sense to add a residual after getting the attended values with 
            #   with the original key/value and then use it as a query in the next timestep  
            #############################################################################################
            # Update query
            if self.use_lstm:
                attended_value, states = self.lstm(attended_value, states)

            query = self.res_ln(self.mlpq(attended_value) + key_value_ln)
            # Append attended queries
            attended_values.append(query.unsqueeze(1))
            if att_weights is not None:
                attention_weights.append(att_weights)
            
        if len(attention_weights):
            attention_weights = torch.cat(attention_weights, dim=0)
        else:
            attention_weights = None

        if self.reduce:
            x3 = self.ln_2(query)
            x4 = self.mlp(x3)
            result = x4 + query
             
        else:
            if len(attended_values):
                attended_values.reverse()
                result = torch.cat(attended_values, axis=1)
                result = self.ln_2(result)
                result = self.mlp(result)

            else: 
                result = query.unsqueeze(1)
                
        
        return result, states, attention_weights
    
    
    
    
    
class BidirectionalTemporalAttention(nn.Module):
    """ Encode a temporal sequence of local features with cross-sequence-attention
        in backward direction
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        reduce: bool=False,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        dropout: float=0.0,
        need_weights: bool=False,
        average_attn_weights: bool=True,
        activation=nn.GELU,
        num_layers: int=1,
        use_lstm: bool=False
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
        
        self.ln_q_backward = norm_layer(hidden_dim)
        self.ln_kv_backward = norm_layer(hidden_dim)
        
        self.mlp_backward = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        self.mlp_forward = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        
        self.ln_q_forward = norm_layer(hidden_dim)
        self.ln_kv_forward = norm_layer(hidden_dim)
        
        self.forward_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.backward_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        
        # MLP block
        self.ln_2 = norm_layer(2*hidden_dim)
        self.mlp = MLPBlock(2*hidden_dim, mlp_dim, activation=activation) 
        self.mlpq = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.use_lstm = use_lstm
        self.res_ln_forward = norm_layer(hidden_dim)
        self.res_ln_backward = norm_layer(hidden_dim)
        
        print("Use LSTM: ", (use_lstm, num_layers))
        
    
    def forward(self, inputs: torch.Tensor, mask: Union[torch.Tensor, None]=None, forward_states: Union[tuple, None]=None, backward_states: Union[tuple, None]=None):
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
        b, n, nf, e = inputs.shape

        # Temporal attention on local features
        attended_values = []
        attention_weights = []
        # Use oldest feature sequence from history as the first query
        query_backward = inputs[:, -1, :, :]        
        query_forward = inputs[:, 0, :, :]   
        att_weights = None
        query = inputs[:, -1, :, :]
        for i in range(1, n):
            # Update key and value
            key_value_forward = inputs[:, i, :, :]
            key_value_backward = inputs[:, n-i-1, :, :]

            # Layer norm 
            query_ln_forward = self.ln_q_forward(query_forward)
            key_value_ln_forward = self.ln_kv_forward(key_value_forward)
            # Layer norm 
            query_ln_backward = self.ln_q_backward(query_backward)
            key_value_ln_backward = self.ln_kv_backward(key_value_backward)
            # query_ln = self.mlpq(query_ln)

            # Backward temporal attention on two subsequent local features at different timestep                   
            forward_attended_value, att_weights = self.forward_attention(
                query_ln_forward, 
                key_value_ln_forward, 
                key_value_ln_forward, 
                need_weights=self.need_weights, 
                average_attn_weights=self.average_attn_weights,
                key_padding_mask=mask[:, i, :] if mask is not None and i < mask.shape[1] else None
            )
            
            if self.use_lstm:
                forward_attended_value, forward_states = self.lstm(query_ln_forward, forward_states)

            # Backward temporal attention on two subsequent local features at different timestep                   
            backward_attended_value, att_weights = self.backward_attention(
                query_ln_backward, 
                key_value_ln_backward, 
                key_value_ln_backward, 
                need_weights=self.need_weights, 
                average_attn_weights=self.average_attn_weights,
                key_padding_mask=mask[:, n-i-1, :] if mask is not None and (n-i-1) < mask.shape[1] else None
            )
            
            if self.use_lstm:
                backward_attended_value, backward_states = self.lstm(query_ln_backward, backward_states)
            
            # attention_weights shape: (1, N, N) where N is temporal sequence length
            #############################################################################################
            # TODO: Think about different ways we could implement/propagate attended values in such a 
            # Backward attention mechanism.
            # - I though maybe it makes sense to add a residual after getting the attended values with 
            #   with the original key/value and then use it as a query in the next timestep  
            #############################################################################################
            # Update query
            

            query_forward = self.res_ln_forward(self.mlp_forward(forward_attended_value) + key_value_ln_forward)
            query_backward = self.res_ln_backward(self.mlp_backward(backward_attended_value) + key_value_ln_backward)
            query = torch.cat([query_forward, query_backward], dim=-1)
            # Append attended queries
            attended_values.append(query.unsqueeze(1))

            if att_weights is not None:
                attention_weights.append(att_weights)
            
        if len(attention_weights):
            attention_weights = torch.cat(attention_weights, dim=0)
        else:
            attention_weights = None

        if self.reduce:
            x3 = self.ln_2(query)
            result = self.mlp(x3)
            result = self.linear(result) 
        else:
            if len(attended_values):
                result = torch.cat(attended_values, axis=1)
                result = self.ln_2(result)
                result = self.mlp(result)
                result = self.linear(result)
            else: 
                result = query.unsqueeze(1)
                
        
        return result, None, attention_weights