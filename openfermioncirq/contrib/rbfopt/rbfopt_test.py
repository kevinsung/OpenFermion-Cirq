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
from openfermioncirq.optimization import (
        BlackBox,
        OptimizationParams,
        OptimizationTrialResult)
from openfermioncirq import (
        HamiltonianObjective,
        VariationalStudy,
        SwapNetworkTrotterAnsatz)
from openfermioncirq.testing import ExampleBlackBox, ExampleBlackBoxNoisy


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
    objective = HamiltonianObjective(hamiltonian)
    study = VariationalStudy('study', ansatz, objective)
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

    trial_result = study.trial_results[0]

    assert isinstance(trial_result, OptimizationTrialResult)
