#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np

from cirq.contrib.model_gradient_descent import model_gradient_descent

from openfermioncirq.optimization import (
    BlackBox,
    OptimizationAlgorithm,
    OptimizationResult)


class ModelGradientDescent(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: np.ndarray=None,
                 initial_guess_array: np.ndarray=None,
                 ) -> OptimizationResult:
        if initial_guess is None:
            raise ValueError('The chosen optimization algorithm requires an '
                             'initial guess.')

        result = model_gradient_descent(
            black_box.evaluate,
            initial_guess,
            sample_radius=self.options.get('sample_radius', 1e-1),
            n_sample_points=self.options.get('n_sample_points', 100),
            rate=self.options.get('rate', 1e-1),
            tol=self.options.get('tol', 1e-8),
            known_values=self.options.get('known_values', None),
            max_evaluations=self.options.get('max_evaluations', None),
            verbose=self.options.get('verbose', False)
        )
        return OptimizationResult(
            optimal_value=result.fun,
            optimal_parameters=result.x,
            num_evaluations=result.nfev
        )

    @property
    def name(self) -> str:
        return 'ModelGradientDescent'
