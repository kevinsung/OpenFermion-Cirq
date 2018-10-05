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

"""CMA-ES"""

from typing import Dict, Optional

import cma
import numpy

from openfermioncirq.optimization import (BlackBox,
                                          OptimizationResult,
                                          OptimizationAlgorithm)


class CmaEs(OptimizationAlgorithm):
    """An optimization algorithm from the scipy.optimize module."""

    def __init__(self,
                 sigma0: float,
                 options: Optional[Dict]=None) -> None:
        self.sigma0 = sigma0
        super().__init__(options)

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None
                 ) -> OptimizationResult:
        result = cma.fmin(black_box.evaluate,
                          initial_guess,
                          self.sigma0,
                          options=self.options)
        return OptimizationResult(optimal_value=result[1],
                                  optimal_parameters=result[0],
                                  num_evaluations=result[2])

    @property
    def name(self) -> str:
        return 'CMA-ES'
