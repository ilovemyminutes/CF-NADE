from logging import getLogger
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.types import ItemId, ItemIndex, UserId, UserRatingHistoryList

logger = getLogger(__name__)

USER_COL_NAME = "user_id"
ITEM_COL_NAME = "movie_id"
RATING_COL_NAME = "rating"


def build_item_indexer(item_ids: List[ItemId]) -> Dict[ItemId, ItemIndex]:
    return {i: idx for idx, i in enumerate(item_ids)}


def ratings_to_user_histories(
    ratings: pd.DataFrame,
    item_indexer: Optional[Dict[ItemId, ItemIndex]],
    user_col_name: str = USER_COL_NAME,
    item_col_name: str = ITEM_COL_NAME,
    rating_col_name: str = RATING_COL_NAME,
) -> Tuple[List[UserId], UserRatingHistoryList]:
    if item_indexer is not None:
        ratings["item_index"] = ratings[item_col_name].map(item_indexer)
        ratings = ratings[ratings["item_index"].notnull()].reset_index(drop=True)
        ratings["item_index"] = ratings["item_index"].astype(int)
        raw_histories = ratings.groupby(user_col_name)[["item_index", rating_col_name]].agg(list)
    else:
        raw_histories = ratings.groupby(user_col_name)[[item_col_name, rating_col_name]].agg(list)
    user_ids = raw_histories.index.tolist()
    user_hists = raw_histories.values.tolist()
    return user_ids, user_hists


class UserHistoryBuilder:
    """
    평점 데이터로부터 다음의 리스트 타입의 유저 히스토리를 추출한다.

    - 입력 데이터 형식:
        |유저ID|아이템ID|평점| 형식의 데이터프레임

    - 출력 데이터 형식:
        UserRatingHistoryList: List[Tuple[Sequence[평가 아이템별 인덱스], Sequence[평가 아이템별 별점]]]
    """

    def __init__(
        self,
        item_ids: List[ItemId],
        user_col_name: str = USER_COL_NAME,
        item_col_name: str = ITEM_COL_NAME,
        rating_col_name: str = RATING_COL_NAME,
    ):
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("item_ids should be composed of unique values.")
        self.item_ids = item_ids
        self.item_indexer = build_item_indexer(item_ids)
        self.col_names = dict(
            user_col_name=user_col_name,
            item_col_name=item_col_name,
            rating_col_name=rating_col_name,
        )

    def build_user_histories(self, ratings: pd.DataFrame) -> Tuple[List[UserId], UserRatingHistoryList]:
        """
        평가 데이터로부터 유저별 평가아이템/평점 시퀀스 리스트를 추출한다.
        """
        ratings = self._deduplicate(ratings)
        user_ids, user_hists = ratings_to_user_histories(ratings, self.item_indexer, **self.col_names)
        return user_ids, user_hists

    def build_user_histories_with_source(
        self, target_ratings: pd.DataFrame, source_ratings: pd.DataFrame
    ) -> Tuple[List[UserId], UserRatingHistoryList, List[UserId], UserRatingHistoryList]:
        """
        타깃/소스 평가 데이터로부터 유저별 평가아이템/평점 시퀀스 리스트를 추출한다.
        """
        target_ratings = self._deduplicate(target_ratings)
        source_ratings = self._deduplicate(source_ratings)
        target_ratings, source_ratings = self._drop_unknowns(target_ratings, source_ratings)

        # sort; synchronize user_id of target and source for indexing
        target_ratings = target_ratings.sort_values(by=USER_COL_NAME)
        source_ratings = source_ratings.sort_values(by=USER_COL_NAME)

        target_user_ids, target_user_hists = ratings_to_user_histories(
            target_ratings, self.item_indexer, **self.col_names
        )
        source_user_ids, source_user_hists = ratings_to_user_histories(
            source_ratings, self.item_indexer, **self.col_names
        )
        return target_user_ids, target_user_hists, source_user_ids, source_user_hists

    @property
    def num_items(self) -> int:
        return len(self.item_ids)

    @staticmethod
    def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임 내 중복된 행을 unique하게 처리한다.
        """
        logger.info("De-duplicate; make duplicated samples unique.")
        pre_size = len(df)
        df = df.drop_duplicates()
        cur_size = len(df)
        if pre_size != cur_size:
            logger.info(f"De-duplicated (# Rows: {pre_size:,d} -> {cur_size:,d})")
        return df

    @staticmethod
    def _drop_unknowns(
        target_ratings: pd.DataFrame,
        source_ratings: pd.DataFrame,
        user_col_name: str = USER_COL_NAME,
        item_col_name: str = ITEM_COL_NAME,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        소스 별점 데이터와 타깃 별점 데이터에 모두 존재하는 유저/콘텐츠를 제외한 모든 데이터(=unknowns)를 버린다.
        * 소스 별점 데이터: 모델의 input에 해당되는 별점 데이터
        * 타깃 별점 데이터: 레이블에 해당하는 별점 데이터
        """
        logger.info("Drop unknown users/items; drop users/items not in both target_ratings and source_ratings")
        for col_name in [user_col_name, item_col_name]:
            source_ids, target_ids = set(source_ratings[col_name]), set(target_ratings[col_name])
            usable_ids = source_ids.intersection(target_ids)

            pre_target_size, pre_source_size = len(target_ratings), len(source_ratings)
            source_ratings = source_ratings[source_ratings[col_name].isin(usable_ids)]
            target_ratings = target_ratings[target_ratings[col_name].isin(usable_ids)]
            cur_target_size, cur_source_size = len(target_ratings), len(source_ratings)

            if pre_target_size > cur_target_size:
                logger.info(
                    f"Dropped unknowns for {col_name} (# Rows of target_ratings: {pre_target_size:,d} ->"
                    f"{cur_target_size:,d})"
                )

            if pre_source_size > cur_source_size:
                logger.info(
                    f"Dropped unknowns for {col_name} (# Rows of source_ratings: {pre_source_size:,d} ->"
                    f"{cur_source_size:,d})"
                )

        return target_ratings, source_ratings
