from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch import optim
from torch.optim import Optimizer

from utils.train_utils import get_optimizer_params
from model.cfnade_loss import CFNADELoss
from model.score_indication import ScoreIndicator
from model.user_encoding import UserEncoder


class CFNADE(pl.LightningModule):
    """
    - Paper: (2016) A Neural Autoregressive Approach to Collaborative Filtering
        - https://arxiv.org/pdf/1605.09477v1.pdf
    - Reference code:
        - (Official) https://github.com/Ian09/CF-NADE
    - NOTE
        - 기존 CFNADE에 ReLU Activation fn., Layer Normalization, Drop-out이 추가되어 있음
    """

    def __init__(
        self,
        m: int,
        k: int,
        h: int,
        j: Optional[int] = None,
        weight_sharing: bool = False,
        dropout: float = 0.5,
        alpha: Optional[float] = 1.0,
        lr: Optional[float] = 2.5e-4,
        weight_decay: Optional[float] = 1e-2,
    ):
        """
        :param m: # contents
        :param k: rating scale
        :param h: hidden dim
        :param weight_sharing: if True, use Param. Sharing of CFNADE
        :param dropout: dropout prob. (default: 0.5)
        :param alpha: (required for train) λ for Ordinal Cost of CFNADE Loss
        :param lr: (required for train) learning rate
        :param weight_decay: (required for train) decay rate for optimization
        """
        super().__init__()
        self.user_encoder = UserEncoder(m, k, h, j, weight_sharing, dropout)
        self.score_indicator = ScoreIndicator(m, k, h, j)
        self.register_buffer("rating_range", torch.arange(1, k + 1))

        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_fn = CFNADELoss(alpha) if alpha is not None else None

    def forward(self, inputs: torch.Tensor, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """NOTE. it is designed for the inference phase
        :param inputs: user rating history data (B x M x K)
        :param indices: if not None, ratings of contents corresponding to input indices will only be predicted.

        :return: B x M or B x LEN(indices)
        """
        h_out = self.user_encoder(inputs)  # B x H
        scores = self.score_indicator(h_out, indices)  # B x M x K (or B x LEN(indices) x K)
        probs = torch.softmax(scores, dim=2)
        pred_ratings = torch.sum(probs * self.rating_range, dim=2)  # B x M (or B x LEN(indices))
        return pred_ratings

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        inputs, labels, seq_lens, seq_split_indices = batch
        h_out = self.user_encoder(inputs)  # B x H
        scores = self.score_indicator(h_out)  # B x M x K
        loss = self.loss_fn(scores, labels, seq_lens, seq_split_indices)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._evaluation_step(batch)

    def test_step(self, batch, batch_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._evaluation_step(batch)

    def validation_epoch_end(self, results: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._evaluation_epoch_end(results, stage="valid")

    def test_epoch_end(self, results: List[Tuple[torch.Tensor, torch.Tensor]]):
        self._evaluation_epoch_end(results, stage="test")

    def configure_optimizers(self) -> Optimizer:
        params_to_optimize = get_optimizer_params(self, weight_decay=self.weight_decay)
        optimizer = optim.AdamW(params_to_optimize, lr=self.lr)
        return optimizer

    def _evaluation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        평가 데이터로부터 얻은 예상별점의 RMSE 측정을 위한 정제
        * 관측값에 대응되는 예상별점만을 남기고 나머지는 제로마스킹
        * M: 전체 콘텐츠 수
        * S: 관측값 수 (=시퀀스 길이)
        """
        inputs, labels = batch
        pred_ratings = self.forward(inputs)

        pred_ratings[labels == 0] = 0
        pred_ratings, labels = pred_ratings.flatten(), labels.flatten()  # M
        pred_ratings, labels = (
            pred_ratings[pred_ratings.nonzero().T.flatten()],
            labels[labels.nonzero().T.flatten()],
        )  # S
        return pred_ratings, labels

    def _evaluation_epoch_end(self, results: List[Tuple[torch.Tensor, torch.Tensor]], stage: str):
        """
        평가 과정 중 _evaluation_step으로 얻은 모든 예상별점/관측값으로 RMSE 측정
        """
        pred_ratings, labels = [], []
        for p, l in results:
            pred_ratings.append(p)
            labels.append(l)
        pred_ratings, labels = torch.cat(pred_ratings), torch.cat(labels)

        # RMSE
        rmse = (labels - pred_ratings).square().mean().sqrt()
        self.log(f"{stage}_rmse", rmse)
