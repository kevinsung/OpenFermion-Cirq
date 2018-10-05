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

from typing import Optional

import numpy
import skopt

from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)


class Forest(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:

        if black_box.bounds is None:
            raise ValueError(
                    'The chosen optimization algorithm requires bounds '
                    'on the black box function inputs.'
            )

        if initial_guess_array is not None:
            x0 = [list(x) for x in initial_guess_array]
        elif initial_guess is not None:
            x0 = list(initial_guess)
        else:
            x0 = None

        result = skopt.forest_minimize(
                black_box.evaluate,
                black_box.bounds,
                x0=x0,
                **self.options)

        return OptimizationResult(optimal_value=result.fun,
                                  optimal_parameters=result.x)


class GBRT(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:

        if black_box.bounds is None:
            raise ValueError(
                    'The chosen optimization algorithm requires bounds '
                    'on the black box function inputs.'
            )

        if initial_guess_array is not None:
            x0 = [list(x) for x in initial_guess_array]
        elif initial_guess is not None:
            x0 = list(initial_guess)
        else:
            x0 = None

        result = skopt.gbrt_minimize(
                black_box.evaluate,
                black_box.bounds,
                x0=x0,
                **self.options)

        return OptimizationResult(optimal_value=result.fun,
                                  optimal_parameters=result.x)


class GaussianProcesses(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:

        if black_box.bounds is None:
            raise ValueError(
                    'The chosen optimization algorithm requires bounds '
                    'on the black box function inputs.'
            )

        if initial_guess_array is not None:
            x0 = [list(x) for x in initial_guess_array]
        elif initial_guess is not None:
            x0 = list(initial_guess)
        else:
            x0 = None

        result = skopt.gp_minimize(
                black_box.evaluate,
                black_box.bounds,
                x0=x0,
                **self.options)

        return OptimizationResult(optimal_value=result.fun,
                                  optimal_parameters=result.x)
