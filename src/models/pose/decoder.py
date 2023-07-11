import torch
import torch.nn as nn

from typing import Union
from functools import partial

from models.vit.mlp import MLPBlock
from models.transformer.encoder import EncoderBlock, Encoder


class PoseDecoderBlock(nn.Module):
    """ Decode a pose keypoint sequence embeddings via Spatial
        Self-Attention
    """

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            dropout: float = 0.0,
            batch_first: bool = True,
            need_weights=False,
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
            out_dim: int,
            dropout: float = 0.0,
            batch_first: bool = True,
            num_layers: int = 4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ) -> None:
        super().__init__()
        self.process_layer = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.decode_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim, dropout=dropout,
                                                       batch_first=batch_first)
        
        self.ln = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim=out_dim)
        # Replace it with custom decoder to allow us to also configure whether to teacher-force during training
        self.decoder = nn.TransformerDecoder(self.decode_layer, num_layers, norm=norm_layer)

        #######################################################
        # TODO: Need to implement a pose decoder block
        # self.block = PoseDecoderBlock(...)
        # ...
        #######################################################

    def generate_square_subsequent_mask(sz: int, device='cpu') -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def process_input(self, img_encoding: torch.Tensor, pos_encoding: torch.Tensor):
        """Processes the image and pose encoding for further use in the decoder

        Args:
            img_encoding(torch.Tensor): Temporal encoding of the image features
            pos_encoding(torch.Tensor): Temporal encoding of the pose

        Returns:
            total_encoding: encoding of image features and pose, transformer based"""

        total_encoding, _ = self.process_layer(img_encoding, pos_encoding, pos_encoding)
        return total_encoding

    def forward(
        self, 
        img_encoding: torch.Tensor, 
        pos_encoding: torch.Tensor,
        tgt: torch.Tensor, 
        is_teacher_forcing: bool=False, 

    ) -> torch.Tensor:
        """Perform forward pass

        Args:
            img_encoding: temporal encoding of the image features
            pos_encoding: temporal encoding of the pose features
            target: target sequence for decoding

        Raises:
            NotImplementedError: Need to implement forward pass
        """
        torch._assert(img_encoding.dim() == 3, f"Expected (batch_size, num_joints, hidden_dim) got {img_encoding.shape}")
        torch._assert(pos_encoding.dim() == 3,
                      f"Expected (batch_size, num_joints, hidden_dim) got {pos_encoding.shape}")

        future_window_plus = tgt.shape[1] # 2nd dim is the sequence length including t=n i.e., future_window + 1
        memory = self.process_input(img_encoding, pos_encoding)
       
        if is_teacher_forcing:
            # No Auto Regression, Pass the tgt as it is, with attention masks to prevent lookahead
            # Generate a (future_window x future_window) matrix mask with -inf at the upper triangular
            # and 0 at the lower. 
            tgt_atten_mask = self.generate_square_subsequent_mask(future_window_plus) 
            result = self.decoder(tgt, memory, tgt_atten_mask=tgt_atten_mask)

        else:
            # Auto Regression, no need for masking
            # Out Shape: (batch_size, 1, num_poses + 1, E)
            # Get the current pose as the initial input to decoder
            tgt_poses = tgt[:, 0, ...].unsueeze(1) 
            for _ in range(future_window_plus - 1): # Loop till future window excluding the pose encoding at t=n
                result = self.decoder(tgt_poses, memory) 
                
                # Concatenate the latest prediction
                tgt_poses = torch.cat([tgt_poses, result[:, -1, ...].unsqueeze(1)], dim=1)
                
        # Linear Projection to out_dim        
        result = self.ln(result)
        result= self.mlp(result)
        return result
