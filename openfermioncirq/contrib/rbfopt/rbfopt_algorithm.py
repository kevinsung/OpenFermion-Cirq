# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import os

import numpy
import rbfopt

from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)


class RBFOptBlackBox(rbfopt.RbfoptBlackBox):

    def __init__(self,
                 black_box: BlackBox,
                 cost_of_evaluate_noisy: Optional[float]=None,
                 confidence_of_evaluate_noisy: Optional[float]=None) -> None:
        self.black_box = black_box
        self.cost_of_evaluate_noisy = cost_of_evaluate_noisy
        self.confidence_of_evaluate_noisy = confidence_of_evaluate_noisy

    def evaluate(self, x: numpy.ndarray) -> float:
        return self.black_box.evaluate(x)

    def evaluate_noisy(self, x: numpy.ndarray) -> numpy.ndarray:
        lower, upper = self.black_box.noise_bounds(
            self.cost_of_evaluate_noisy, self.confidence_of_evaluate_noisy)
        return numpy.array([
            self.black_box.evaluate_with_cost(x, self.cost_of_evaluate_noisy),
            lower,
            upper])

    def get_dimension(self) -> int:
        return self.black_box.dimension

    def get_var_lower(self) -> numpy.ndarray:
        return numpy.array([bound[0] for bound in self.black_box.bounds])

    def get_var_upper(self) -> numpy.ndarray:
        return numpy.array([bound[1] for bound in self.black_box.bounds])

    def get_var_type(self) -> numpy.ndarray:
        return numpy.array(['R'] * self.get_dimension())

    def has_evaluate_noisy(self) -> bool:
        return self.cost_of_evaluate_noisy is not None


class RBFOpt(OptimizationAlgorithm):

    def __init__(self,
                 options: Optional[Dict]=None,
                 cost_of_evaluate_noisy: Optional[float]=None,
                 confidence_of_evaluate_noisy: Optional[float]=None) -> None:
        self.cost_of_evaluate_noisy = cost_of_evaluate_noisy
        self.confidence_of_evaluate_noisy = confidence_of_evaluate_noisy
        super().__init__(options)

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None,
                 output_stream_path=os.devnull
                 ) -> OptimizationResult:
        if black_box.bounds is None:
            raise ValueError(
                    'The chosen algorithm requires bounds on the function '
                    'arguments.')
        rbfopt_settings = rbfopt.RbfoptSettings(**self.options)
        rbfopt_black_box = RBFOptBlackBox(black_box,
                                          self.cost_of_evaluate_noisy,
                                          self.confidence_of_evaluate_noisy)
        if initial_guess_array is None:
            initial_guess_array = [initial_guess]

        if initial_guess_array is not None or initial_guess is not None:
            init_node_val = numpy.array(
                    [black_box.evaluate(initial_guess_array[i])
                     for i in range(len(initial_guess_array))])
        else:
            init_node_val = None

        algorithm = rbfopt.RbfoptAlgorithm(
            rbfopt_settings,
            rbfopt_black_box,
            init_node_pos=initial_guess_array,
            init_node_val=init_node_val)
        with open(output_stream_path, 'w') as output_stream:
            algorithm.set_output_stream(output_stream)
            algorithm.optimize()
        return OptimizationResult(
                optimal_value=algorithm.fbest,
                optimal_parameters=algorithm.all_node_pos[
                    algorithm.fbest_index],
                num_evaluations=algorithm.evalcount)
