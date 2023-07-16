"""
Code was originally taken from PyTorch.

"""

import torch
import torch.nn as nn

from typing import Union
from functools import partial
from collections import OrderedDict

from models.vit.encoder import Encoder


class VisionTransformer(nn.Module):
    """ Vision Transformer as per https://arxiv.org/abs/2010.11929.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        total_layers: int,
        need_weights: bool=False,
        global_pool: str="avg",
    ):
        super().__init__()
        torch._assert(num_layers < total_layers, "The number of layers to use cannot be larger than the total number of layers")
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = 1000

        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        kernel_size = patch_size
        stride = patch_size

        # Add a class token
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=kernel_size, stride=stride
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        ## The entire encoder
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            total_layers,
            self.norm_layer,
            need_weights,
        )
        self.seq_length = seq_length
        self.global_pool = global_pool
        # Need to define it to be able to load the state dictionary.
        # This head wont be used.
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, self.num_classes)
        self.heads = nn.Sequential(heads_layers)

        self.max_pool = nn.MaxPool2d(kernel_size=[patch_size, patch_size], stride=patch_size)
        # ##############################################################################
        # # TODO: Need to add two new heads for local and global feature extraction
        # # ... 
        # # ...
            
        
        # raise NotImplementedError

    def _process_padding_mask(self, padding_mask):
        
        padding_mask = torch.flatten(self.max_pool(padding_mask), start_dim=1, end_dim=-1)
        
        return padding_mask.bool()
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """ Given an (N, C, H, W) image tensor, it returns an (N, S, E) tensor of tokens,
            where N is batch size, S is number of tokens, and E is length of each token.
        """

        n, c, h, w  = x.shape
        p = self.patch_size

        # Make sure the input size is what we're prepared for!
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x, n_h, n_w

    def forward(self, x: torch.Tensor, key_padding_mask: Union[torch.Tensor, None]):
        """_summary_

        Args:
            x (torch.Tensor): Input (B*N, C, H, W) tensor

        Returns:
            local (torch.Tensor): An output tensor of shape (B, N, H) with N as 
                the number of patches with H as the hidden dimension.
                
            Global (torch.Tensor): An output tensor of shape (B, H) with H as the 
                hidden dimension.
        """
        torch._assert(key_padding_mask.shape[0] == x.shape[0], f"The first dimension must be equal. Got {x.shape[0]} and {key_padding_mask.shape[0]}")
        torch._assert((key_padding_mask.shape[2] == x.shape[2] and key_padding_mask.shape[3] == x.shape[3]), 
                      f"The height and width dimension must be equal. Got ({x.shape[2]}, {x.shape[3]}) and ({key_padding_mask.shape[2]}, {key_padding_mask.shape[3]})")
        
        # Reshape and permute the input tensor
        x, n_h, n_w = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)

        # Add the CLS token
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Convert 2D padding mask to 1D vector of bool (True where there is padding else False)
        # Also add additonal padding mask for cls token
        key_padding_mask = self._process_padding_mask(key_padding_mask)
        cls_mask = torch.zeros(size=(key_padding_mask.shape[0], 1))
        cls_mask = cls_mask > 0
        key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=-1)
        results = self.encoder(x, key_padding_mask)

        # Take out the CLS token (in fact "tokens" because we have a batch)
        cls_token = results[:, 0]
        
        ##################################################################
        # TODO: Need to call the new head on all outputs including cls
        # ... 
        # ...
        
        local_feat = results # (B, 197, 768)
        if self.global_pool == "avg":
            global_feat = results.mean(dim=1) # (B, 768)
            
        elif self.global_pool == "max":
            global_feat = torch.max(results, dim=1) # (B, 768)
            
        return local_feat, global_feat

    
    def visualize_cls(self, attention_weights, n_h, n_w):
        r"""
            Parameters:
            attention_weights: Tensor(N, S+1, S+1), where N is batch size, S is number of tokens (+1 for CLS token).
            It assumes that CLS token is the first token in both 2nd and 3rd dimensions

            n_h, n_w: int, original Height and width of the tokenized input (before putting all tokens along each other).
            Normally S should be equal to n_h * n_w

            Returns:
            Tensor(N, n_h, n_w, 1): a 2D attention map of the CLS token for each sample.
        """
        ######################################
        # TODO: Maybe required later for viz
        # ...
        ######################################
        
        # For individual heads
        attention_weights = attention_weights[:, :, 0, 1:].reshape(-1, 12, n_h, n_w)

        # If to visualize the attention weights averaged over heads
        # attention_weights = attention_weights[:, :, 0, 1:].reshape(-1, 12, n_h, n_w).sum(dim=1) / 12
        # attention_weights = attention_weights.squeeze(1)


        return attention_weights
        
        