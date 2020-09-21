from typing import List

from abstract.perceptron import AbstractPerceptron


class Perceptron(AbstractPerceptron):
    def _train(self, laps: int):
        for _ in range(laps):
            for i, sample in enumerate(self._input_data):

                neuron_status = self._activation_F(self._SUM(sample))

                if not neuron_status:
                    self._backward_propagation(sample, self._output_data[i], neuron_status)

    def test_data_sample(self, dataset: List[list]):
        return [self._activation_F(self._SUM(d)) for d in dataset]
