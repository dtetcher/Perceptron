from typing import List

from abstract.perceptron import AbstractPerceptron
from concrete.checkers import WeightsAnalyser


class Perceptron(AbstractPerceptron):
    def _train(self, laps: int):

        analyzer = WeightsAnalyser(len(self._input_data))

        for lap in range(laps):
            for i, sample in enumerate(self._input_data):

                neuron_status = self._activation_F(self._SUM(sample))
                if not neuron_status:
                    self._backward_propagation(sample, self._output_data[i], neuron_status)

                analyzer.push(self.synaptic_weights)
                if analyzer.is_stable():
                    return lap

        return lap

    def test_data_sample(self, dataset: List[list]):
        return [self._activation_F(self._SUM(d)) for d in dataset]
