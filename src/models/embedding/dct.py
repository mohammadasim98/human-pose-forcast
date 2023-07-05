###############################################
# TODO: Implement DCT encoding
###############################################

import torch
from torch import nn
import math
import base_encoding


class FourEncoding(base_encoding.BaseEncoding):
    """Applies positional encoding to the input based on discrete fourier transform"""

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
            for i in range(self.d_model):
                cos_val = math.cos((math.pi/self.seq_length)*(i + .5)*(k + .5))
                for j in range(self.batch_size):
                    encoding_matrix[j, k, i] = cos_val
        return encoding_matrix

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO: not sure if this is what you planned, I understood the idea of DCT encoding in this way
        return input * self.encoding_matrix
