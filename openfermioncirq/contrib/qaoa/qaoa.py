import numpy
import cirq

from openfermioncirq.variational import VariationalAnsatz
from openfermioncirq.variational.letter_with_subscripts import LetterWithSubscripts

class QAOAMaxCutAnsatz(VariationalAnsatz):

    def __init__(self,
                 graph,
                 p,
                 adiabatic_evolution_time=None,
                 qubits=None
                 ):
        self.graph = graph
        self.p = p

        if adiabatic_evolution_time is None:
            adiabatic_evolution_time = float(graph.size())
        self.adiabatic_evolution_time = adiabatic_evolution_time

        super().__init__(qubits)

    def params(self):
        for i in range(self.p):
            yield LetterWithSubscripts('gamma', i)
            yield LetterWithSubscripts('beta', i)

    def param_bounds(self):
        bounds = []
        for param in self.params():
            bounds.append((-1.0, 1.0))
        return bounds

    def _generate_qubits(self):
        return cirq.LineQubit.range(len(self.graph))

    def operations(self, qubits):
        yield cirq.H.on_each(*qubits)
        for i in range(self.p):
            gamma = LetterWithSubscripts('gamma', i)
            beta = LetterWithSubscripts('beta', i)
            yield (cirq.ZZPowGate(exponent=gamma).on(qubits[j], qubits[k])
                   for j, k in self.graph.edges)
            yield cirq.XPowGate(exponent=beta).on_each(*qubits)

    def default_initial_params(self):
        total_time = self.adiabatic_evolution_time
        step_time = total_time / self.p

        params = []
        for i in range(self.p):
            interpolation_progress = 0.5 * (2 * i + 1) / self.p
            params.append(_canonicalize_exponent(
                -interpolation_progress * step_time / numpy.pi, 2))
            params.append(_canonicalize_exponent(
                (1-interpolation_progress) * step_time / numpy.pi, 2))

        return numpy.array(params)


def _canonicalize_exponent(exponent: float, period: int) -> float:
    # Shift into [-p/2, +p/2).
    exponent += period / 2
    exponent %= period
    exponent -= period / 2
    # Prefer (-p/2, +p/2] over [-p/2, +p/2).
    if exponent <= -period / 2:
        exponent += period  # coverage: ignore
    return exponent
