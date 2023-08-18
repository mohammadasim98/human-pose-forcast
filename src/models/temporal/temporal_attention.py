import torch
import torch.nn as nn

from functools import partial
from typing import Union
from models.vit.mlp import MLPBlock 
from models.common.sequential import AttentionMultiInputSequential 


class BasicAttentionBlock(nn.Module):
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
        need_weights: bool=False,
        average_attn_weights: bool=True,
        activation=nn.GELU,
        query_residual: bool=True

    ) -> None:
        
        super().__init__()
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        self.ln_1 = norm_layer(hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        
        self.query_residual = query_residual
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Union[torch.Tensor, None]=None):
        
        att_weights = None
        
        attended_value, att_weights = self.attention(
            query, 
            key, 
            value, 
            need_weights=True, 
            average_attn_weights=True,
            key_padding_mask=mask
        )

        attended_value = attended_value + query
        attended_value = self.ln_1(attended_value)
        
        attended_value_mlp = self.mlp(attended_value)
        
        attended_value_mlp += attended_value
        attended_value_ln = self.ln_2(attended_value_mlp)
        result = attended_value_ln


        return result, att_weights

        

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
        use_lstm: bool=False,
        num_lstm_layers: int=3,
        num_query: int=18

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
                
#         self.ln_q = norm_layer(hidden_dim)
#         self.ln_kv = norm_layer(hidden_dim)
        layers = []
        for i in range(num_layers):
            layers.append(
                BasicAttentionBlock(
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    norm_layer=norm_layer,
                    dropout=dropout, 
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    activation=activation,
                    query_residual=True if i < num_layers-1 else False
                )
            )
        
        self.layers = AttentionMultiInputSequential(*layers)
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layers, batch_first=True)
        self.res_ln = norm_layer(hidden_dim)
        self.use_lstm = use_lstm

        self.query = nn.Parameter(torch.randn(1, num_query, hidden_dim))
        
        self.query_mask = torch.zeros(1, num_query).bool().to("cuda:0")

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
        query = self.query.expand(b, -1, -1)
        query_mask = self.query_mask.expand(b, -1)
        # attended_values.append(query.unsqueeze(1))
        att_weights = None
        # query_ln = self.ln_q(query)            
        for i in range(0, hw):
            # Update query
            key_value = inputs[:, i, :, :]
            state = key_value
            # Layer norm 
            # key_value_ln = self.ln_kv(key_value)
            # query_ln = self.mlpq(query_ln)
            # key_value_cat = torch.cat([key_value, query], dim=1)
            # Forward temporal attention on two subsequent local features at different timestep  
            # attended_value, att_weights = self.layers(
            #     query=query, 
            #     key=key_value, 
            #     value=key_value, 
            #     mask=torch.cat([mask[:, i, :], query_mask], dim=-1) if mask is not None and i < mask.shape[1] else None
            # )
            
            attended_value, att_weights = self.layers(
                query=query, 
                key=key_value, 
                value=key_value, 
                mask=mask[:, i, :] if mask is not None and i < mask.shape[1] else None
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
            
            # query = self.res_ln(attended_value + key_value)
            query = attended_value
            
            # Append attended queries
            attended_values.append(query.unsqueeze(1))
            if len(att_weights):
                attention_weights.append(att_weights)

        if len(attention_weights):
            attention_weights = attention_weights
        else:
            attention_weights = None
        
        if self.reduce:
            result = self.mlp(query)
            result += query
            result_ln = self.ln_2(result)
        
        else:
            if len(attended_values):    
                queries = torch.cat(attended_values, axis=1)
                result = self.mlp(queries)
                result += queries
                result_ln = self.ln_2(result)

            else: 
                query = query.unsqueeze(1)
                result = self.mlp(query)
                result += query
                result_ln = self.ln_2(result)
        
        return result_ln, states, attention_weights


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
        use_lstm: bool=False,
        num_lstm_layers: int=3,
        num_query: int=18
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
        
#         self.ln_q = norm_layer(hidden_dim)
#         self.ln_kv = norm_layer(hidden_dim)
        
        layers = []
        for i in range(num_layers):
            layers.append(
                BasicAttentionBlock(
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    norm_layer=norm_layer,
                    dropout=dropout, 
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    activation=activation,
                    query_residual=True if i < num_layers-1 else False
                )
            )
        
        self.layers = AttentionMultiInputSequential(*layers)
        
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        
        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, activation=activation) 
        
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layers, batch_first=True)        
        
        self.use_lstm = use_lstm
        self.res_ln = norm_layer(hidden_dim)
        self.query = nn.Parameter(torch.randn(1, num_query, hidden_dim))
        self.query_mask = torch.zeros(1, num_query).bool().to("cuda:0")

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
        query = self.query.expand(b, -1, -1) 
        query_mask = self.query_mask.expand(b, -1) 
        # attended_values.append(query.unsqueeze(1))
        # query_ln = self.ln_q(query)
        att_weights = None
        for i in range(hw-1, -1, -1):
            # Update key and value
            key_value = inputs[:, i, :, :]

            # Layer norm 
            # key_value_ln = self.ln_kv(key_value)
            
            # query_ln = self.mlpq(query_ln)
            # key_value_cat = torch.cat([key_value, query], dim=1)

            # Backward temporal attention on two subsequent local features at different timestep                   
            # attended_value, att_weights = self.layers(
            #     query=query, 
            #     key=key_value, 
            #     value=key_value,
            #     mask=torch.cat([mask[:, i, :], query_mask], dim=-1) if mask is not None and i < mask.shape[1] else None
            # )
            
            attended_value, att_weights = self.layers(
                query=query, 
                key=key_value, 
                value=key_value, 
                mask=mask[:, i, :] if mask is not None and i < mask.shape[1] else None
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

            # query = self.res_ln(attended_value + key_value)
            query = attended_value
            # Append attended queries
            attended_values.append(query.unsqueeze(1))
            if len(att_weights):
                attention_weights.append(att_weights)
            
        if len(attention_weights):
            attention_weights = torch.cat(attention_weights, dim=0)
        else:
            attention_weights = None

        if self.reduce:
            result = self.mlp(query)
            result += query
            result_ln = self.ln_2(result)
        
        else:
            if len(attended_values): 
                attended_values.reverse()
                queries = torch.cat(attended_values, axis=1)
                result = self.mlp(queries)
                result += queries
                result_ln = self.ln_2(result)

            else: 
                query = query.unsqueeze(1)
                result = self.mlp(query)
                result += query
                result_ln = self.ln_2(result)  
        
        return result_ln, states, attention_weights
    
    
    
    
    
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
        use_lstm: bool=False,
        num_lstm_layers: int=1,
        num_query: int=18
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
        
#         self.ln_q_backward = norm_layer(hidden_dim)
#         self.ln_kv_backward = norm_layer(hidden_dim)

        
#         self.ln_q_forward = norm_layer(hidden_dim)
#         self.ln_kv_forward = norm_layer(hidden_dim)
        
        layers = []
        for i in range(num_layers):
            layers.append(
                BasicAttentionBlock(
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    norm_layer=norm_layer,
                    dropout=dropout, 
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    activation=activation,
                    query_residual=True if i < num_layers-1 else False
                )
            )
        
        self.forward_attention = AttentionMultiInputSequential(*layers)
        
        layers = []
        for i in range(num_layers):
            layers.append(
                BasicAttentionBlock(
                    num_heads=num_heads, 
                    hidden_dim=hidden_dim, 
                    mlp_dim=mlp_dim,
                    norm_layer=norm_layer,
                    dropout=dropout, 
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                    activation=activation,
                    query_residual=True if i < num_layers-1 else False
                )
            )
        
        self.backward_attention = AttentionMultiInputSequential(*layers)
        
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        
        # MLP block
        self.ln_2 = norm_layer(2*hidden_dim)
        self.mlp = MLPBlock(2*hidden_dim, mlp_dim, activation=activation) 
        if use_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layers, batch_first=True)
        self.linear = nn.Linear(2*hidden_dim, hidden_dim)
        
        self.use_lstm = use_lstm
        self.res_ln_forward = norm_layer(hidden_dim)
        self.res_ln_backward = norm_layer(hidden_dim)
        
        self.query_forward = nn.Parameter(torch.randn(1, num_query, hidden_dim))
        self.query_backward = nn.Parameter(torch.randn(1, num_query, hidden_dim))
        self.query_mask = torch.zeros(1, num_query).bool().to("cuda:0")

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
        attention_weights = None
        # Temporal attention on local features
        forward_attended_values = []
        backward_attended_values = []
        attention_weights = []
        # Use oldest feature sequence from history as the first query
        query_backward = self.query_backward.expand(b, -1, -1)     
        query_forward = self.query_forward.expand(b, -1, -1)
        query_mask = self.query_mask.expand(b, -1)
        att_weights = None
        query = inputs[:, -1, :, :]
        for i in range(0, n):
            # Update key and value
            key_value_forward = inputs[:, i, :, :]
            key_value_backward = inputs[:, n-i-1, :, :]

            # Layer norm 
            # query_ln_forward = self.ln_q_forward(query_forward)
            # key_value_ln_forward = self.ln_kv_forward(key_value_forward)
            # # Layer norm 
            # query_ln_backward = self.ln_q_backward(query_backward)
            # key_value_ln_backward = self.ln_kv_backward(key_value_backward)
            # query_ln = self.mlpq(query_ln)
            # key_value_forward_cat = torch.cat([key_value_forward, query_forward], dim=1)
            # key_value_backward_cat = torch.cat([key_value_backward, query_backward], dim=1)
            # Backward temporal attention on two subsequent local features at different timestep                   
            # forward_attended_value, att_weights = self.forward_attention(
            #     query=query_forward, 
            #     key=key_value_forward, 
            #     value=key_value_forward, 
            #     mask=torch.cat([mask[:, i, :], query_mask], dim=-1) if mask is not None and i < mask.shape[1] else None
            # )
            
            forward_attended_value, att_weights = self.forward_attention(
                query=query_forward, 
                key=key_value_forward, 
                value=key_value_forward, 
                mask=mask[:, i, :] if mask is not None and i < mask.shape[1] else None
            )
            
            if self.use_lstm:
                forward_attended_value, forward_states = self.lstm(query_ln_forward, forward_states)

            # Backward temporal attention on two subsequent local features at different timestep                   
            # backward_attended_value, att_weights = self.backward_attention(
            #     query=query_backward, 
            #     key=key_value_backward, 
            #     value=key_value_backward, 
            #     mask=torch.cat([mask[:, n-i-1, :], query_mask], dim=-1) if mask is not None and (n-i-1) < mask.shape[1] else None
            # )
            backward_attended_value, att_weights = self.backward_attention(
                query=query_backward, 
                key=key_value_backward, 
                value=key_value_backward, 
                mask=mask[:, n-i-1, :] if mask is not None and (n-i-1) < mask.shape[1] else None
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
            

            # query_forward = self.res_ln_forward(forward_attended_value + key_value_forward)
            # query_backward = self.res_ln_backward(backward_attended_value + key_value_backward)
            
            query_forward = forward_attended_value
            query_backward = backward_attended_value
            # query = torch.cat([query_forward, query_backward], dim=-1)
            # Append attended queries
            forward_attended_values.append(query_forward.unsqueeze(1))
            backward_attended_values.append(query_backward.unsqueeze(1))

            # if len(att_weights):
            #     attention_weights.append(att_weights)
            
        # if len(attention_weights):
        #     attention_weights = torch.cat(attention_weights, dim=0)
        # else:
        #     attention_weights = None
        
        
        if self.reduce:
            if len(forward_attended_values) and len(backward_attended_values):
                query =  torch.cat([query_forward, query_backward], dim=-1)
            result = self.mlp(query)
            result += query
            result_ln = self.ln_2(result)
            result_ln = self.linear(result_ln)

        else:
            if len(forward_attended_values) and len(backward_attended_values): 
                backward_attended_values.reverse()
                
                queries_forward = torch.cat(forward_attended_values, axis=1)
                queries_backward = torch.cat(backward_attended_values, axis=1)
                
                queries =  torch.cat([queries_forward, queries_backward], dim=-1)
                result = self.mlp(queries)
                result += queries
                result_ln = self.ln_2(result)
                result_ln = self.linear(result_ln)

            else: 
                query = query.unsqueeze(1)
                result = self.mlp(query)
                result += query
                result_ln = self.ln_2(result)  
                
        
        return result_ln, None, attention_weights