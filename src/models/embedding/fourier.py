###############################################
# TODO: Implement fourier encoding
###############################################

import torch
from torch import nn
import math

class FourierEncoding(nn.Module):
    """Applies positional encoding to the input as described in 'Attention is all you need'"""

    def __int__(self,
                batch_size: int,
                seq_length: int,
                d_model: int,
                n: int = 10000):
        super().__init__(batch_size, seq_length, d_model, n)
        self.encoding_matrix = self.build_encoding_matrix()

    def build_encoding_matrix(self):
        encoding_matrix = torch.empty((self.batch_size, self.seq_length, self.d_model))
        for k in range(self.seq_length):
            for i in range(int(self.d_model / 2)):
                d = self.n ** (2 * i / self.d_model)
                sin_val = math.sin(k / d)
                cos_val = math.cos(k / d)
                for j in range(self.batch_size):
                    encoding_matrix[j, k, 2 * i] = sin_val
                    encoding_matrix[j, k, 2 * i + 1] = cos_val
        return encoding_matrix

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return input + self.encoding_matrix

class FourierMLPEncoding(nn.Module):
    """
    Inspired from positional encodings in NeRF
    """

    def __init__(self, num_freq, d_model, n_input_dim, activation=nn.GELU):
        super().__init__()
        self.P_embed = num_freq
        self.d_model = d_model
        self.n_input_dim = n_input_dim
        self.pose_encoding_size = self.n_input_dim * (1 + 2 * num_freq)
        self.ff = nn.Sequential(
            nn.Linear(self.pose_encoding_size, self.d_model // 2),
            activation(),
            nn.Linear(self.d_model // 2, self.d_model),
        )

    def forward(self, position):
        rets = [position]
        for i in range(self.P_embed):
            for fn in [torch.sin, torch.cos]:
                rets.append(fn(2.0**i * position))
        out = torch.cat(rets, dim=-1)
        return self.ff(out)