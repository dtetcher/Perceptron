import abc


class ActivationFunction(abc.ABC):
    @abc.abstractmethod
    def activate(self, _sum: float) -> float:
        pass
