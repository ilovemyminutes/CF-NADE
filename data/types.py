from typing import Sequence, Tuple, List

UserId = int

ItemId = int

ItemIndex = int

Rating = int

UserRatingHistoryList = List[Tuple[Sequence[ItemIndex], Sequence[Rating]]]
