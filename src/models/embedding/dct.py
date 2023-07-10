


###############################################
# TODO: Implement DCT encoding 
###############################################


import torch
import torch.nn as nn
import torch_dct as dct



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