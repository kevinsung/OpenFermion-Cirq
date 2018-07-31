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
import poap.controller
import pySOT

from openfermioncirq.optimization import (
        BlackBox,
        OptimizationAlgorithm,
        OptimizationResult)


class PySOTBlackBox:

    def __init__(self,
                 black_box: BlackBox) -> None:
        if black_box.bounds is None:
            raise ValueError('A PySOT algorithm requires bounds.')
        self.black_box = black_box

    @property
    def dim(self) -> int:
        return self.black_box.dimension

    @property
    def xlow(self) -> numpy.ndarray:
        return numpy.array([bound[0] for bound in self.black_box.bounds])

    @property
    def xup(self) -> numpy.ndarray:
        return numpy.array([bound[1] for bound in self.black_box.bounds])

    @property
    def integer(self) -> numpy.ndarray:
        return numpy.array([])

    @property
    def continuous(self) -> numpy.ndarray:
        return numpy.arange(self.dim)

    def objfunction(self, x: numpy.ndarray) -> float:
        return self.black_box.evaluate(x)


class Dycors(OptimizationAlgorithm):

    def optimize(self,
                 black_box: BlackBox,
                 initial_guess: Optional[numpy.ndarray]=None,
                 initial_guess_array: Optional[numpy.ndarray]=None,
                 ) -> OptimizationResult:
        if black_box.bounds is None:
            raise ValueError(
                    'The chosen algorithm requires bounds on the function '
                    'arguments.')

        maxeval = self.options['maxeval']

        # (1) Optimization problem
        data = PySOTBlackBox(black_box)

        # (2) Experimental design
        # Use a symmetric Latin hypercube with 2d + 1 samples
        exp_des = pySOT.SymmetricLatinHypercube(
                dim=data.dim,
                npts=2*data.dim+1)

        # (3) Surrogate model
        # Use a cubic RBF interpolant with a linear tail
        surrogate = pySOT.RBFInterpolant(
                kernel=pySOT.CubicKernel,
                tail=pySOT.LinearTail,
                maxp=maxeval)

        # (4) Adaptive sampling
        # Use DYCORS with 100d candidate points
        adapt_samp = pySOT.CandidateDYCORS(
                data=data,
                numcand=100*data.dim)

        # Use the serial controller (uses only one thread)
        controller = poap.controller.SerialController(data.objfunction)

        # (5) Use the sychronous strategy without non-bound constraints
        strategy = pySOT.SyncStrategyNoConstraints(
                worker_id=0,
                data=data,
                maxeval=maxeval,
                nsamples=1,
                exp_design=exp_des,
                response_surface=surrogate,
                sampling_method=adapt_samp)

        controller.strategy = strategy

        # Run the optimization strategy
        result = controller.run()

        return OptimizationResult(
                optimal_value=result.value,
                optimal_parameters=result.params[0])
