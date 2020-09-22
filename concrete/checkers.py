from copy import deepcopy
from typing import List


class Check:

    @staticmethod
    def if_input_valid(dataset: List[list], size: int, return_dset_size: bool = False):
        if len(dataset) == 0:
            raise ValueError(f"Dataset can't be empty")

        for data in dataset:
            if len(data) != size:
                raise ValueError(f"Dataset size isn't equal to desired value - {size}")

        if return_dset_size: return size


class WeightsAnalyser:
    def __init__(self, dataset_size: int):
        self.__dataset_size = dataset_size
        self.__weights_storage: List[list] = list()

    def is_stable(self) -> bool:

        if len(self.__weights_storage) == self.__dataset_size:
            status = self.__is_equal()

            self.__weights_storage.clear()
            return status

        return False

    def __is_equal(self) -> bool:
        first_set = self.__weights_storage[0]

        for d_set in self.__weights_storage:
            if d_set != first_set:
                return False
        return True

    def push(self, d_set: List):
        self.__weights_storage.append(deepcopy(d_set))
