import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from data.types import UserRatingHistoryList, UserId


class TrainRatingDataset(Dataset):
    def __init__(
        self,
        user_histories: UserRatingHistoryList,
        num_items: int,
        rating_scale: int = 10,
        random_ordering: bool = True,
    ):
        super().__init__()
        self.user_histories = user_histories
        self.num_items = num_items
        self.rating_scale = rating_scale
        self.random_ordering = random_ordering

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        indices, values = self.user_histories[idx]
        indices = torch.tensor(indices, dtype=torch.long)
        values = torch.tensor(values, dtype=torch.long)

        # random ordering
        seq_len: int = len(indices)  # 유저가 실제로 평가한 아이템 수 (논문의 D)
        seq_split_idx: int = random.choice(range(1, seq_len))  # 시퀀스 내 input/label을 분기할 지점(=인덱스)
        seq_order = np.arange(seq_len)
        if self.random_ordering:
            seq_order = np.random.permutation(seq_order)

        input_flags, label_flags = seq_order < seq_split_idx, seq_order >= seq_split_idx
        input_indices, input_values = indices[input_flags], values[input_flags]
        label_indices, label_values = indices[label_flags], values[label_flags]

        # input
        input_ = torch.zeros(self.num_items, self.rating_scale)
        input_[input_indices, input_values - 1] = 1  # onehot

        # label
        label = torch.zeros(self.num_items, self.rating_scale)
        label[label_indices, label_values - 1] = 1  # onehot

        return input_, label, seq_len, seq_split_idx

    def __len__(self):
        return len(self.user_histories)


class EvalRatingDataset(Dataset):
    def __init__(
        self,
        input_user_histories: UserRatingHistoryList,
        label_user_histories: UserRatingHistoryList,
        num_items: int,
        rating_scale: int,
    ):
        super().__init__()
        self.input_user_histories = input_user_histories
        self.label_user_histories = label_user_histories
        self.num_items = num_items
        self.rating_scale = rating_scale

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # input
        input_indices, input_values = self.input_user_histories[idx]
        input_indices = torch.tensor(input_indices, dtype=torch.long)
        input_values = torch.tensor(input_values, dtype=torch.long)

        input_ = torch.zeros(self.num_items, self.rating_scale)
        input_[input_indices, input_values - 1] = 1  # make ratings onehot

        # label (NOTE: NOT apply onehot)
        label_indices, label_values = self.label_user_histories[idx]
        label_indices = torch.tensor(label_indices, dtype=torch.long)
        label_values = torch.tensor(label_values, dtype=torch.float)

        label = torch.zeros(self.num_items, dtype=torch.float)
        label[label_indices] = label_values

        return input_, label

    def __len__(self):
        return len(self.label_user_histories)


class InferRatingPredDataset(Dataset):
    def __init__(
        self,
        user_ids: List[UserId],
        user_histories: UserRatingHistoryList,
        num_items: int,
        rating_scale: int = 10,
    ):
        super().__init__()
        self.user_ids = user_ids
        self.user_histories = user_histories

        if len(self.user_ids) != len(self.user_histories):
            raise ValueError("the length of user_ids must be the same as that of user_histories.")
        self.num_items = num_items
        self.rating_scale = rating_scale

    def __getitem__(self, idx: int) -> Tuple[UserId, Dict[str, torch.Tensor]]:
        user_id: UserId = self.user_ids[idx]
        input_indices, input_values = self.user_histories[idx]
        input_indices = torch.tensor(input_indices, dtype=torch.long)
        input_values = torch.tensor(input_values, dtype=torch.long)

        input_ = torch.zeros(self.num_items, self.rating_scale)
        input_[input_indices, input_values - 1] = 1  # make ratings onehot
        model_input = dict(inputs=input_)
        return user_id, model_input

    def __len__(self):
        return len(self.user_ids)
