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

from typing import Any, List, Optional, Sequence, Tuple

import numpy
from qiskit.tools.apps.optimization import SPSA_optimization

from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)

from openfermioncirq.testing import (
        ExampleAnsatz,
        ExampleVariationalObjective)


class Spsa(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:

        def bb(x):
            return black_box.evaluate(x), [None]

        spsa_parameters = (
            self.options['a'],
            self.options['c'],
            self.options['alpha'],
            self.options['gamma'],
            self.options['A']
        )

        (f_res, x_res, _, _, _, _,), _ = SPSA_optimization(
            bb,
            initial_guess,
            spsa_parameters,
            self.options['max_evaluations']//2)

        return OptimizationResult(
            optimal_value=f_res,
            optimal_parameters=x_res,
            num_evaluations=None)
