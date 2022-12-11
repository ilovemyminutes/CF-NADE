import torch
from torch import nn


class CFNADELoss(nn.Module):
    """
    Reference.
    * (2016) A Neural Autoregressive Approach to Collaborative Filtering
    * https://github.com/Ian09/CF-NADE
    * https://github.com/JoonyoungYi/CFNADE-keras
    """

    def __init__(self, alpha: float = 1.0, reduction: str = "mean", eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha  # λ in paper
        if reduction not in ["sum", "mean"]:
            raise NotImplementedError("'reduction' must be one of 'mean' and 'sum'")
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_split_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param scores: B(batch size) x M(# contents) x K(rating scale)
        :param labels: B x M x K
        :param seq_lens: B
        :param seq_split_indices: B
        :return: an aggregated loss (scalar)
        """
        # + self.eps; prevent NaN loss caused by torch.log(x~0)
        probs = torch.softmax(scores, dim=2) + self.eps  # B x M x K

        cost_regular = self._calculate_regular_cost(probs, labels)
        cost_ordinal = self._calculate_ordinal_cost(probs, labels)

        cost = (1 - self.alpha) * cost_regular.sum(dim=1) + self.alpha * cost_ordinal.sum(dim=1)
        cost = (seq_lens / (seq_lens - seq_split_indices + self.eps)) * cost  # B
        cost = cost.sum() if self.reduction == "sum" else cost.mean()
        return cost

    @staticmethod
    def _calculate_regular_cost(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cost_regular = -torch.sum(labels * torch.log(probs), dim=2)  # B x M
        return cost_regular

    @staticmethod
    def _calculate_ordinal_cost(probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # cumulative probability
        cumul_probs_up = torch.cumsum(probs, dim=2)  # rating from k to K
        cumul_probs_down = torch.cumsum(probs.flip(dims=[2]), dim=2).flip(dims=[2])  # rating from k to 1

        mask_up = torch.cumsum(labels.flip(dims=[2]), dim=2).flip(dims=[2])
        mask_down = torch.cumsum(labels, dim=2)

        # ordinal cost
        cost_ordinal_up = -torch.sum((torch.log(probs) - torch.log(cumul_probs_up)) * mask_up, dim=2)
        cost_ordinal_down = -torch.sum((torch.log(probs) - torch.log(cumul_probs_down)) * mask_down, dim=2)
        cost_ordinal = cost_ordinal_up + cost_ordinal_down  # B x M
        return cost_ordinal
