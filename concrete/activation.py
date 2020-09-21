from abstract.activation import ActivationFunction


class Threshold(ActivationFunction):
    def activate(self, _sum: float):
        return float(_sum > 1)
