import networkx
import cirq
import openfermion

from openfermioncirq.contrib.qaoa import QAOAMaxCutAnsatz


def test_ansatz():

    # Set problem parameters
    n = 6
    p = 2

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Generate qubits
    qubits = cirq.LineQubit.range(n)

    # Create ansatz
    ansatz = QAOAMaxCutAnsatz(graph, p, qubits)

    print(ansatz.circuit)



test_ansatz()
