from typing import List

from abstract.activation import ActivationFunction as AFunc
from abstract.perceptron import AbstractPerceptron
from concrete.checkers import WeightsAnalyser


class Perceptron(AbstractPerceptron):

    def __init__(self, input_data: List[list],
                 output_layer_pattern,
                 activation_function: AFunc,
                 stuck_condition: bool = True,
                 min_weight_val: float = -0.5,
                 max_weight_val: float = 0.5,
                 training_speed: float = 0.05,
                 config_file: str = './dump.json',
                 ):

        self.__stuck_condition = stuck_condition

        super().__init__(input_data, output_layer_pattern, activation_function,
                         min_weight_val, max_weight_val, training_speed, config_file)

    def _train(self, laps: int, ):

        analyzer = WeightsAnalyser(len(self._input_data))

        for lap in range(laps):
            for i, sample in enumerate(self._input_data):

                neuron_status = self._activation_F(self._SUM(sample))
                if not neuron_status:
                    self._backward_propagation(sample, self._output_data[i], neuron_status)

                if self.__stuck_condition:
                    analyzer.push(self.synaptic_weights)
                    if analyzer.is_stable():
                        return lap

        return laps

    def test_data_sample(self, dataset: List[list]):
        return [self._activation_F(self._SUM(d)) for d in dataset]
