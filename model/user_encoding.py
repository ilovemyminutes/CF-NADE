from typing import Optional

import torch
from torch import nn

from model import initialize


class UserEncoder(nn.Module):
    def __init__(
        self,
        m: int,
        k: int,
        h: int,
        j: Optional[int] = None,
        weight_sharing: bool = True,
        dropout: float = 0.5,
    ):
        """
        :param m: # contents
        :param k: rating scale
        :param h: hidden dim
        :param j: factorization factor
        :param weight_sharing: if True, use Param. Sharing of CFNADE
        :param dropout: dropout prob. (default: 0.5)
        """
        super().__init__()
        if j is None:
            self._W = nn.Parameter(torch.rand(m, k, h))
            self._A = None
            self._B = None
            initialize(self._W)
        else:
            self._W = None
            self._A = nn.Parameter(torch.rand(k, m, j))
            self._B = nn.Parameter(torch.rand(j, h))
            initialize(self._A)
            initialize(self._B)
        self.norm = nn.LayerNorm(h)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sharing = weight_sharing

    def forward(self, x: torch.Tensor):
        """
        :notation:
            B: batch size
            M: # items for item-based, # users for user-based model
            K: rating scale
        :param x: B x M x K
        :return: B x H
        """
        if self.sharing:
            x = torch.cumsum(x.flip(dims=[2]), dim=2).flip(dims=[2])
        h_out = torch.tensordot(x, self.W, dims=[[1, 2], [0, 1]])  # B x H
        h_out = self.norm(h_out)
        h_out = self.act(h_out)
        h_out = self.dropout(h_out)
        return h_out

    @property
    def W(self) -> torch.Tensor:
        """
        :return: W; M x K x H
        """
        if self._W is not None:
            return self._W
        return torch.matmul(self._A, self._B).transpose(0, 1)
