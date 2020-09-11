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
