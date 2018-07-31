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

import openfermion

from openfermioncirq.contrib import Dycors
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationParams,
        OptimizationTrialResult)
from openfermioncirq import (
        HamiltonianObjective,
        VariationalStudy,
        SwapNetworkTrotterAnsatz)


class ExampleBlackBox(BlackBox):

    @property
    def dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return [(-2.0, 2.0), (-2.0, 2.0)]

    def _evaluate(self,
                  x: numpy.ndarray) -> float:
        return numpy.sum(x**2)


def test_dycors_optimize():
    black_box = ExampleBlackBox()
    algorithm = Dycors(options={'maxeval': 5})
    result = algorithm.optimize(black_box)

    assert isinstance(result.optimal_value, float)


def test_dycors_optimize_study():
    grid = openfermion.Grid(2, 2, 1.0)
    jellium = openfermion.jellium_model(grid, spinless=True, plane_wave=False)
    hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(jellium)

    ansatz = SwapNetworkTrotterAnsatz(hamiltonian)
    objective = HamiltonianObjective(hamiltonian)
    study = VariationalStudy('study', ansatz, objective)
    algorithm = Dycors(options={'maxeval': 3 * study.num_params})

    study.optimize(
            OptimizationParams(
                algorithm,
                cost_of_evaluate=1e4),
            reevaluate_final_params=True)

    result = study.results[0]
    assert isinstance(result, OptimizationTrialResult)
