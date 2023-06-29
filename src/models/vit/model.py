"""
Code was originally taken from PyTorch.

"""
import torch
import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from torchvision.ops.misc import MLP

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=0.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):

        for i in range(2):
            for type in ["weight", "bias"]:
                old_key = f"{prefix}linear_{i+1}.{type}"
                new_key = f"{prefix}{3*i}.{type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
        need_weights = False,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim)

        self.need_weights = need_weights # Whether to return attention weights as well

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        ################################
        # TODO Q1: Implement the encoder block based on  what was in Lecture 7 slide 46.
        # Note that all the relevant modules are already defined in the init above.
        # You may also have to return attention weights based on self.need_weights
        # Hint: There's already a `need_weights` argument for nn.MultiheadAttention forward pass. See the docs.

        result = None
        attention_weights = None # Needed only if self.need_weights is True for this specific Block

        x0 = self.ln_1(input)
        x1, attention_weights = self.self_attention(x0, x0, x0, need_weights=self.need_weights)
        x2 = input + x1
        x3 = self.ln_2(x2)
        x4 = self.mlp(x3)
        result = x2 + x4
        ################################

        if self.need_weights:
            return result, attention_weights
        else: 
            return result


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        norm_layer = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            ##################################
            ## TODO Q1: Stack up the EncoderBlocks (simply initialize them)
            ## Please only modify the right side with an instance of EncoderBlock
            ## If you change the names of layers then the pre-trained weights will not load!
            ## Hint: Set need_weights to True only for last Block and False for others
            ## As we would like to visualize attention weights for last layer later.

            layers[f"encoder_layer_{i}"] = EncoderBlock(num_heads, hidden_dim, mlp_dim, norm_layer,
                                                        need_weights=True if i == num_layers - 1 else False)
            # Replace None with Block
            #################################

        self.layers = nn.Sequential(layers)
        
        # final layer norm
        self.ln = norm_layer(hidden_dim) 

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        #####################################
        ### TODO Q1: Apply the forward pass over the Encoder.
        ## 1. Add Positional Embedding (self.pos_embedding) to the input
        ## 2. Feed it to self.layers and get the result and attention_weights
        input += self.pos_embedding
        result, attention_weights = self.layers(input)
        #####################################

        result = self.ln(result) # Final layer norm

        return result, attention_weights


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_classes = 1000

        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        ############################
        # TODO Q1: put the correct kernel_size and stride to apply on the image
        # in order to get (patch_size x patch_size) non-overlapping patches.
        # For example, in the figure in lecture, the image is broken 
        # into 9 non-overlapping patches.
        kernel_size = patch_size
        stride = patch_size

        ############################
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
            self.norm_layer,
        )
        self.seq_length = seq_length


        # Final classification head
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["head"] = nn.Linear(hidden_dim, self.num_classes)
        self.heads = nn.Sequential(heads_layers)

    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an (N, C, H, W) image tensor, it returns an (N, S, E) tensor of tokens,
        where N is batch size, S is number of tokens, and E is length of each token.
        """

        n, c, h, w = x.shape
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

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x, n_h, n_w = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)

        # Add the CLS token
        x = torch.cat([batch_class_token, x], dim=1)
        
        ##########################
        # TODO Q1: Feed it to the encoder and get the results and attention_weights
        results, attention_weights = self.encoder(x)

        ##########################

        # Take out the CLS token (in fact "tokens" because we have a batch)
        cls_token = results[:, 0]
        
        ##########################
        # TODO Q1: Apply the final classification head
        final_logits = self.heads(cls_token)
        final_logits = nn.Softmax(dim=1)(final_logits)

        ##########################

        visualized_attention = self.visualize_cls(attention_weights, n_h, n_w)
        return final_logits, visualized_attention
    
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
        #########################
        # TODO Q2: Return a 2D attention map.
        # Note that the given attention_weights are for all tokens at the final Encoder blocks
        # and are of shape (N, self.seq_length+1,self.seq_length+1), where N is the batch size.
        # You have to take out only the attention for CLS token with respect to other tokens
        # and reshape it into a 2D grid.
        # Hint: Go a few lines above to see how to take the attention token.
        # Also note that the attention of CLS token w.r.t other tokens also includes
        # its attention to itself (which should not be used for visualization)

        # For individual heads
        attention_weights = attention_weights[:, :, 0, 1:].reshape(-1, 12, n_h, n_w)

        # If to visualize the attention weights averaged over heads
        # attention_weights = attention_weights[:, :, 0, 1:].reshape(-1, 12, n_h, n_w).sum(dim=1) / 12
        # attention_weights = attention_weights.squeeze(1)


        return attention_weights
        #########################