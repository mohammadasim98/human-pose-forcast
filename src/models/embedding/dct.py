


###############################################
# TODO: Implement DCT encoding 
###############################################


import torch
import torch.nn as nn
import torch_dct as dct

import numpy as np


class DiscreteCosineTransformEmbedding(nn.Module):
    """ Pose Encoder
    """

    def __init__(
        self,
        ) -> None:

        super().__init__()

    def forward(self, x: torch.Tensor):
        """ Performs DCT transform

        Args:
            x (torch.tensor): A (B, N, J, 3) of each 2D pose keypoints.

        Returns:
            out (torch.tensor): A (B, N, J, 3) embedding of each 2D pose keypoints.
        """
        torch._assert(x.dim() == 4, f"Expected (batch_size, history_window, num_joints, 3) got {x.shape}")

        x = x.permute(0, 3, 2, 1)

        out = dct.dct(x)
        out = out.permute(0, 3, 2, 1)

        return out

class InverseDiscreteCosineTransformEmbedding(nn.Module):
    """ Pose Encoder
    """

    def __init__(
        self,
        ) -> None:

        super().__init__()

    def forward(self, x: torch.Tensor):
        """ Performs Inverse DCT transform

        Args:
            x (torch.tensor): A (B, N, J, 3) of each 2D pose keypoints.

        Returns:
            out (torch.tensor): A (B, N, J, 3) embedding of each 2D pose keypoints.
        """
        torch._assert(x.dim() == 4, f"Expected (batch_size, history_window, num_joints, 3) got {x.shape}")

        x = x.permute(0, 3, 2, 1)

        out = dct.idct(x)
        out = out.permute(0, 3, 2, 1)

        return out
    
    
def get_dct_matrix(N):
    dct_m = torch.eye(N)
    for k in torch.arange(N):
        for i in torch.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * torch.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = torch.linalg.inv(dct_m)
    return dct_m, idct_m
        
        