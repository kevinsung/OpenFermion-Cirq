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
import pytest

from openfermioncirq.contrib.model_gradient_descent import ModelGradientDescent
from openfermioncirq.testing import ExampleBlackBox


def test_mgd():
    black_box = ExampleBlackBox()
    initial_guess = numpy.ones(black_box.dimension)
    algorithm = ModelGradientDescent(
            options={
                'sample_radius': 1e-1,
                'n_sample_points': 100,
                'rate': 1e-1,
                'tol': 1e-8,
                'max_evaluations': 10000
            }
    )
    result = algorithm.optimize(black_box, initial_guess)

    assert isinstance(result.optimal_value, float)
    assert isinstance(result.optimal_parameters, numpy.ndarray)
    assert isinstance(result.num_evaluations, int)


def test_mgd_requires_initial_guess():
    black_box = ExampleBlackBox()
    algorithm = ModelGradientDescent()
    with pytest.raises(ValueError):
        _ = algorithm.optimize(black_box)

def test_mgd_name():
    algorithm = ModelGradientDescent()
    assert ModelGradientDescent().name == 'ModelGradientDescent'
