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

import numpy

from openfermioncirq import VariationalStudy
from openfermioncirq.contrib import Nomad
from openfermioncirq.optimization import (
        OptimizationParams,
        OptimizationTrialResult)
from openfermioncirq.testing import (
        ExampleAnsatz,
        ExampleBlackBox,
        ExampleVariationalObjective)


def test_nomad_optimize():
    black_box = ExampleBlackBox()
    algorithm = Nomad({'MAX_BB_EVAL': 10})

    result = algorithm.optimize(black_box, initial_guess=numpy.zeros(2))

    assert isinstance(result.optimal_value, float)