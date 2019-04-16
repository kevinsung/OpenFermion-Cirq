import networkx
import numpy

import cirq
import openfermion

from openfermioncirq.contrib.qaoa import QAOAMaxCutAnsatz


def test_ansatz():

    # Set problem parameters
    n = 10
    p = 100

    # Generate a random 3-regular graph on n nodes
    graph = networkx.random_regular_graph(3, n)

    # Generate qubits
    qubits = cirq.LineQubit.range(n)

    # Create ansatz
    ansatz = QAOAMaxCutAnsatz(graph, p, None, qubits)

    # Create Hamiltonian
    hamiltonian = openfermion.QubitOperator()
    for i, j in graph.edges:
        hamiltonian += openfermion.QubitOperator(((i, 'Z'), (j, 'Z')))

    hamiltonian_sparse = openfermion.get_sparse_operator(hamiltonian)
    ground_energy, _ = openfermion.get_ground_state(hamiltonian_sparse)

    circuit = ansatz.circuit
    num_params = len(list(ansatz.params()))

    zeros = numpy.zeros(num_params)
    zeros_circuit = cirq.resolve_parameters(circuit, ansatz.param_resolver(zeros))
    zeros_state = zeros_circuit.apply_unitary_effect_to_state()
    zeros_val = openfermion.expectation(hamiltonian_sparse, zeros_state)

    adiabatic = ansatz.default_initial_params()
    adiabatic_circuit = cirq.resolve_parameters(circuit, ansatz.param_resolver(adiabatic))
    adiabatic_state = adiabatic_circuit.apply_unitary_effect_to_state()
    adiabatic_val = openfermion.expectation(hamiltonian_sparse, adiabatic_state)
    
    print(zeros_val)
    print(adiabatic_val)
    print(ground_energy)


test_ansatz()
