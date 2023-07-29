import torch
import torch.nn as nn

from typing import Union
from functools import partial

from models.vit.mlp import MLPBlock
from models.transformer.encoder import EncoderBlock, Encoder
from models.temporal.encoder import TemporalEncoder

class PoseDecoderV2(nn.Module):
    """ Pose Encoder
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            out_dim: int,
            temporal: dict,
            img_dim: int,
            pose_dim: int,
            num_layers: int = 4,
            dropout: float = 0.0,
            batch_first: bool = True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_global: bool=False,
            need_weights: bool=False,
            average_attn_weights: bool=True,
            residual: bool=False,
            activation = nn.GELU
    ) -> None:
        super().__init__()
        self.dual_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)

        self.decode_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, dropout=dropout,
                                                       batch_first=batch_first, activation=activation())
        
        self.ln = norm_layer(hidden_dim)
        self.ln2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim=out_dim, activation=activation)
        # Replace it with custom decoder to allow us to also configure whether to teacher-force during training
        self.decoder = nn.TransformerDecoder(self.decode_layer, num_layers)
        self.temporal_encoder = TemporalEncoder(**temporal["encoder"], use_global=use_global, activation=activation)

        self.need_weights=need_weights,
        self.average_attn_weights=average_attn_weights
        self.residual=residual
        #######################################################
        # TODO: Need to implement a pose decoder block
        # self.block = PoseDecoderBlock(...)
        # ...
        #######################################################

    def _generate_square_subsequent_mask(sz: int, device='cuda:0') -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def process_local_input(self, img_encoding: torch.Tensor, pos_encoding: torch.Tensor):
        """Processes the image and pose encoding for further use in the decoder

        Args:
            img_encoding(torch.Tensor): Temporal encoding of the image features
            pos_encoding(torch.Tensor): Temporal encoding of the pose

        Returns:
            total_encoding: encoding of image features and pose, transformer based"""
        

        attention_weights = None
        total_encoding, attention_weights = self.dual_local_attention(
            img_encoding, 
            pos_encoding, 
            pos_encoding, 
            need_weights=self.need_weights,
            average_attn_weights=self.average_attn_weights
        )
        return total_encoding, attention_weights
    
    
    def process_global_input(self, img_encoding: torch.Tensor, pos_encoding: torch.Tensor):
        """Processes the image and pose encoding for further use in the decoder

        Args:
            img_encoding(torch.Tensor): Temporal encoding of the image features
            pos_encoding(torch.Tensor): Temporal encoding of the pose

        Returns:
            total_encoding: encoding of image features and pose, transformer based"""
        
        

        attention_weights = None
        total_encoding, attention_weights = self.dual_global_attention(
            img_encoding, 
            pos_encoding, 
            pos_encoding, 
            need_weights=self.need_weights,
            average_attn_weights=self.average_attn_weights
        )
        return total_encoding, attention_weights

    def forward(
        self, 
        img_encoding: torch.Tensor, 
        pos_encoding: torch.Tensor,
        future_window: int=15,
        history_window: int=15,
        is_teacher_forcing: bool=False,
        future: Union[torch.Tensor, None]=None

    ) -> torch.Tensor:
        """Perform forward pass

        Args:
            img_encoding: temporal encoding of the image features
            pos_encoding: temporal encoding of the pose features
            target: target sequence for decoding

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(img_encoding.dim() == 3, f"Expected (batch_size, history_window + num_patches + 1, hidden_dim) got {img_encoding.shape}")
        torch._assert(pos_encoding.dim() == 3, f"Expected (batch_size, num_joints + history_window, hidden_dim) got {pos_encoding.shape}")
        torch._assert(tgt.dim() == 4, f"Expected (B, future_window + 1, num_joints + 1, hidden_dim) got {pos_encoding.shape}")
        future_window_plus = future_window + 1
        
        cross_attention_weights = None
        
        img_encoding_local = img_encoding[:, history_window:, ...]
        pos_encoding_local = pos_encoding[:, history_window:, ...]
        
        img_encoding_global = img_encoding[:, :history_window, ...]
        pos_encoding_global = pos_encoding[:, :history_window, ...]
        
        # (B, 18, 256)
        local_memory, local_cross_attention_weights = self.process_local_input(img_encoding_local, pos_encoding_local)
        
        # (B, 15, 256)
        global_memory, global_cross_attention_weights = self.process_global_input(img_encoding_global, pos_encoding_global)
        
        memory, cross_attention_weights = self.local_global_attention(global_memory, local_memory, local_memory)

        # Auto Regression, no need for masking
        # Out Shape: (batch_size, 1, num_poses + 1, E)
        # Get the current pose as the initial input to decoder
        
        
        inp_poses = tgt[:, -1, ...].unsqueeze(1)
        tgt_poses = tgt[:, -1, ...].unsqueeze(1)
        if is_teacher_forcing:
            future_poses = torch.cat([inp_poses, future], dim=1)
        for i in range(future_window_plus - 1): # Loop till future window excluding the pose encoding at t=n
            
                
            tgt_poses_temp, _, temp_attention_weights = self.temporal_encoder(
                local_feat=inp_poses, 
            )      

            result = self.decoder(tgt=tgt_poses_temp, memory=memory) 
            result = self.ln(result)
            result= self.mlp(result)
            
            if self.residual:
                result += inp_poses[:, -1, ...]
            
            # Concatenate the latest prediction
            tgt_poses = torch.cat([tgt_poses, result.unsqueeze(1)], dim=1)
            
            if is_teacher_forcing:
                inp_poses = future_poses[:, :i+2, ...]
            else:
                inp_poses = tgt_poses
                
        # Linear Projection to out_dim        
        tgt_poses = self.ln2(tgt_poses)

        return tgt_poses, [local_cross_attention_weights, global_cross_attention_weights, temp_attention_weights]