import random
from abc import ABC, abstractmethod
from typing import TypeVar, Generator

T = TypeVar("T")


class Dataloader(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> T: ...

    @classmethod
    def __permutation(cls, length: int, shuffle: bool):
        identity_permutation = list(range(length))
        if shuffle:
            random.shuffle(identity_permutation)
        return identity_permutation

    def iterate(
        self, batch_size: int = 1, shuffle: bool = False
    ) -> Generator[T, None, None]:
        dataset_len = self.__len__()
        iter_permutation = self.__permutation(dataset_len, shuffle)
        start, end = 0, min(batch_size, dataset_len)
        while start < dataset_len:
            if batch_size == 1:
                yield self.__getitem__(iter_permutation[start])
            else:
                batch = []
                for item in self.__getitem__(iter_permutation[start]):
                    batch.append([item])
                for idx in range(start + 1, end):
                    for item_idx, item in enumerate(
                        self.__getitem__(iter_permutation[idx])
                    ):
                        batch[item_idx].append(item)
                yield tuple(batch)
            start, end = end, min(end + batch_size, dataset_len)
