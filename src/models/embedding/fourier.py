###############################################
# TODO: Implement fourier encoding
###############################################

import torch
from torch import nn
import math

class FourEncoding(nn.Module):
    """Applies positional encoding to the input as described in 'Attention is all you need'"""

    def __int__(self,
                batch_size: int,
                seq_length: int,
                d_model: int,
                n: int = 10000):
        super().__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.n = n
        self.batch_size = batch_size
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
