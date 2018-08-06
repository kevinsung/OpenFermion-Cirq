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

from typing import Optional, Sequence, Tuple

import numpy
import pytest

from openfermioncirq import VariationalStudy
from openfermioncirq.contrib import Forest, GBRT, GaussianProcesses
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationParams,
        OptimizationTrialResult)
from openfermioncirq.testing import (
        ExampleAnsatz,
        ExampleVariationalObjective)


class ExampleBlackBox(BlackBox):
    """Returns the sum of the squares of the inputs.

    skopt requires the black box inputs to be lists instead of numpy arrays,
    so we can't use the ExampleBlackBox from the testing module
    """

    @property
    def dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return [(-10.0, 10.0), (-10.0, 10.0)]

    def _evaluate(self,
                  x: numpy.ndarray) -> float:
        return sum(a**2 for a in x)


forest_algorithm = Forest(options={'n_calls': 11})
GBRT_algorithm = GBRT(options={'n_calls': 11})
gaussian_processes_algorithm = GaussianProcesses(options={'n_calls': 11})


@pytest.mark.parametrize('algorithm', [
    forest_algorithm,
    GBRT_algorithm,
    gaussian_processes_algorithm])
def test_skopt_algorithms_optimize(algorithm):
    black_box = ExampleBlackBox()
    result = algorithm.optimize(black_box, initial_guess=numpy.zeros(2))

    assert isinstance(result.optimal_value, float)
