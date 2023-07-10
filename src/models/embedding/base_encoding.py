import torch
from torch import nn
import math


class BaseEncoding(nn.Module):
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
        raise NotImplementedError

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
