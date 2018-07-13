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
import pybobyqa

from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)


class Bobyqa(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        bounds = None
        if black_box.bounds is not None:
            mins = [bound[0] for bound in black_box.bounds]
            maxs = [bound[1] for bound in black_box.bounds]
            bounds = (numpy.array(mins), numpy.array(maxs))
        result = pybobyqa.solve(black_box.evaluate,
                                initial_guess,
                                bounds=bounds,
                                **self.options)
        return OptimizationResult(optimal_value=result.f,
                              optimal_parameters=result.x,
                              num_evaluations=result.nf)
