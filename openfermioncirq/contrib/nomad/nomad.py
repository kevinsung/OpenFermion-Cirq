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
import PyNomad

from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)

from openfermioncirq.testing import (
        ExampleAnsatz,
        ExampleVariationalObjective)


class Nomad(OptimizationAlgorithm):
    
    def __init__(self, options: Optional[Any]=None):
        self.params = ['BB_OUTPUT_TYPE OBJ']
        options = options or {}
        for key, val in options.items():
            self.params.append('{} {}'.format(key, val))
        super().__init__(options)

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:

        if black_box.bounds is None:
            raise ValueError(
                    'The chosen algorithm requires bounds on the function '
                    'arguments.')

        mins = [bound[0] for bound in black_box.bounds]
        maxs = [bound[1] for bound in black_box.bounds]
        
        def bb(x):
            dim = x.get_n()
            z = numpy.array([x.get_coord(i) for i in range(dim)])
            x.set_bb_output(0, black_box.evaluate(z))
            return 1

        x_res, f_res, h, nb_evals, nb_iters, stopflag = PyNomad.optimize(
            bb, list(initial_guess), mins, maxs, self.params)

        return OptimizationResult(
            optimal_value=f_res,
            optimal_parameters=numpy.array(x_res),
            num_evaluations=nb_evals)
