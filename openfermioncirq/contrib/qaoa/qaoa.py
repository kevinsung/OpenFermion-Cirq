import cirq

from openfermioncirq.variational import VariationalAnsatz
from openfermioncirq.variational.letter_with_subscripts import LetterWithSubscripts

class QAOAMaxCutAnsatz(VariationalAnsatz):

    def __init__(self,
                 graph,
                 p,
                 adiabatic_evolution_time=None,
                 qubits
                 ):
        self.graph = graph
        self.p = p

        if adiabatic_evolution_time is None:
            adiabatic_evolution_time = float(graph.size())

        super().__init__(qubits)

    def params(self):
        for i in range(p):
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
