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

from openfermioncirq.contrib import RBFOpt
from openfermioncirq.optimization import BlackBox, OptimizationTrialResult
from openfermioncirq import (HamiltonianVariationalStudy,
                             OptimizationParams,
                             SwapNetworkTrotterAnsatz)


class ExampleBlackBox(BlackBox):

    @property
    def dimension(self) -> int:
        return 2

    @property
    def bounds(self) -> Optional[Sequence[Tuple[float, float]]]:
        return [(-2.0, 2.0), (-2.0, 2.0)]

    def evaluate(self,
                 x: numpy.ndarray) -> float:
        return numpy.sum(x**2)


class ExampleBlackBoxNoisy(ExampleBlackBox):

    def evaluate_with_cost(self,
                           x: numpy.ndarray,
                           cost: float) -> float:
        return numpy.sum(x**2) + 1 / cost

    def noise_bounds(self,
                     cost: float,
                     confidence: Optional[float]=None
                     ) -> Tuple[float, float]:
        return -2 * confidence / cost, 2 * confidence / cost


def test_rbfopt_optimize():
    black_box = ExampleBlackBoxNoisy()
    algorithm = RBFOpt(
            options={'max_evaluations': 5,
                     'max_noisy_evaluations': 1},
            cost_of_evaluate_noisy=1e6,
            confidence_of_evaluate_noisy=.99)
    result = algorithm.optimize(
            black_box,
            initial_guess_array=numpy.array(
                [[0.1, 0.1],
                 [-0.1, -0.1]]))

    assert isinstance(result.optimal_value, float)


def test_rbfopt_optimize_study():
    grid = openfermion.Grid(2, 2, 1.0)
    jellium = openfermion.jellium_model(grid, spinless=True, plane_wave=False)
    hamiltonian = openfermion.get_diagonal_coulomb_hamiltonian(jellium)

    ansatz = SwapNetworkTrotterAnsatz(hamiltonian)
    study = HamiltonianVariationalStudy('study', ansatz, hamiltonian)
    algorithm = RBFOpt(
            options={'max_evaluations': 10,
                     'max_noisy_evaluations': 5},
            cost_of_evaluate_noisy=1e6,
            confidence_of_evaluate_noisy=.99)

    study.optimize(
            OptimizationParams(
                algorithm,
                cost_of_evaluate=1e4),
            reevaluate_final_params=True)

    result = study.results[0]
    assert isinstance(result, OptimizationTrialResult)
