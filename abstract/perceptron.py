from typing import List
import random as rd
import json
import abc

from abstract.activation import ActivationFunction as AFunc
from concrete.checkers import Check


class AbstractPerceptron(abc.ABC):
    def __init__(self, input_data: List[list],          # List of train data
                 output_layer_pattern,                  # Lambda pattern which used for output layer data generation
                 activation_function: AFunc,        # Neural network activation function
                 min_weight_val: float = -0.5,          # Minimal and maximal synaptic weight's values
                 max_weight_val: float = 0.5,           # # # # Used for weights generation
                 training_speed: float = 0.05,          # Speed of neural network training
                 config_file: str = './dump.json',      # Path to .json file with training data
                 ):

        self._dataset_size = Check.if_input_valid(input_data,
                                                  len(input_data[0]),
                                                  return_dset_size=True,
                                                  )
        self._config = config_file
        self._input_data = input_data
        self._output_data = self._generate_output_layer(output_layer_pattern)
        self._activation_F = activation_function.activate

        self.__min_w = min_weight_val
        self.__max_w = max_weight_val
        self.__training_speed = training_speed

        self._synaptic_weights = [rd.uniform(min_weight_val, max_weight_val)
                                  for _ in range(self._dataset_size)]

    @abc.abstractmethod
    def _train(self, laps: int) -> None:
        pass

    def begin_training(self, laps: int = 20000, flush: bool = False):
        print("Starting training....\n\n")

        if not flush:
            print("Statistics before training",
                  self.statistics(),
                  sep=f"\n{20 * '-'}\n")

        laps = self._train(laps)

        print(f"Training finished({laps} laps).\n\n")

        if not flush:
            print("Statistics after training",
                  self.statistics(),
                  sep=f"\n{20 * '|'}\n")

            print("{:<10}{:>10}".format("Current", "Expected"))

            for idx in range(len(self._input_data)):
                print("{:<10}{:>10}".format(
                    int(self._activation_F(self._SUM(self._input_data[idx]))),
                    self._output_data[idx])
                )

    def _backward_propagation(self, data_sample: list, t: int, y: float):
        error = t - y

        for i, unit in enumerate(data_sample):
            adjustment = self.__training_speed * error * unit
            self._synaptic_weights[i] += adjustment

    def _SUM(self, data_sample: list):
        _sum = 0
        for i in range(len(data_sample)):
            _sum += self._synaptic_weights[i] * data_sample[i]
        return _sum

    @abc.abstractmethod
    def test_data_sample(self, data_sample: list):
        pass

    def _generate_output_layer(self, expression):
        __out_data = []
        for input_row in self._input_data:
            r = int(expression(input_row))
            __out_data.append(r)
        return __out_data

    def dump(self, flush: bool = False):
        print(f"Saving data to {self._config}...") if not flush else None
        self._dump_config()
        print("OK") if not flush else None

    def _dump_config(self):
        data_hash = {
            'SynapticWeights': self._synaptic_weights,
        }

        with open(self._config, 'wt') as f:
            json.dump(data_hash, f, ensure_ascii=False, indent=4)

    def load(self):
        self._load_config()

    def _load_config(self):
        with open(self._config, 'rt') as f:
            conf: dict = json.load(f)

        self._synaptic_weights = conf['SynapticWeights']

    def statistics(self):
        return str(f"Weights: {self._synaptic_weights}\n"
                   f"Min: {self.__min_w}\n"
                   f"Max: {self.__max_w}\n"
                   f"SpeedWeight: {self.__training_speed}\n")

    @property
    def synaptic_weights(self):
        return self._synaptic_weights

