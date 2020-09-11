from abstract.a_activation_function import AbstractActivationFunction


class Threshold(AbstractActivationFunction):
    def activate(self, _sum: float):
        return float(_sum > 1)
