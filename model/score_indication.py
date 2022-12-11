from typing import Optional

import torch
from torch import nn

from model import initialize


class ScoreIndicator(nn.Module):
    def __init__(self, m: int, k: int, h: int, j: Optional[int] = None):
        """
        :param m: # contents
        :param k: rating scale
        :param h: hidden dim
        :param j: factorization factor
        """
        super().__init__()
        if j is None:
            self._V = nn.Parameter(torch.rand(h, m, k))
            self._P = None
            self._Q = None
            self.b = nn.Parameter(torch.rand(m, k))
            initialize(self._V, self.b)
        else:
            self._V = None
            self._P = nn.Parameter(torch.rand(k, m, j))
            self._Q = nn.Parameter(torch.rand(j, h))
            self.b = nn.Parameter(torch.rand(m, k))
            # 초기화 절차: (1) P, Q 초기화  (2) 초기화된 P, Q로 임시적으로 V를 생성, 해당 분포를 참고해 b 초기화
            # NOTE. initialize() 내부적으로 V를 다시 한번 초기화하는 과정이 포함되어 엄밀히 위 (2) 절차에 이슈가 있는데,
            # 실험적으로 아래와 같이 초기화 하는 게 더 좋은 성능을 보여 아래 형태를 유지함.
            initialize(self._P)
            initialize(self._Q)
            initialize(self.V, self.b)

    def forward(self, h_out: torch.Tensor, indices: Optional[torch.Tensor] = None):
        """
        :notation:
            B: batch size
            M: # items for item-based, # users for user-based model
            K: rating scale

        :param h_out: B x H
        :param indices: if not None, scores of contents corresponding to input indices will only be output.
        :return: B x M x K (or B x LEN(indices) x K)
        """
        if indices is None:
            scores = torch.tensordot(h_out, self.V, dims=[[1], [0]]) + self.b  # B x M x K
        else:
            if indices.ndim == 0:
                indices = indices.unsqueeze(0)  # e.g. tensor(1) -> tensor([1])
            scores = torch.tensordot(h_out, self.V[:, indices, :], dims=[[1], [0]]) + self.b[indices, :]
        scores = torch.cumsum(scores, dim=2)  # B x M x K (or B x LEN(indices) x K)
        return scores

    @property
    def V(self) -> nn.Parameter:
        """
        :return: V; H x M x K
        """
        if self._V is not None:
            return self._V
        return torch.matmul(self._P, self._Q).transpose(0, 2)
