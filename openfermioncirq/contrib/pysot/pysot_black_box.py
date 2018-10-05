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

import numpy

from openfermioncirq.optimization import BlackBox


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
