import abc


class AbstractActivationFunction(abc.ABC):
    @abc.abstractmethod
    def activate(self, _sum: float):
        pass
