## These file will contain all the ansatze used for variational quantum simulation

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


#=========================================

def hweff_ansatz(p):
	n_spins = 3
	count = 0
	circuit = QuantumCircuit(n_spins)
	depth = 3

	for j in range(depth):

		if(j%2 == 0):
			# Rx - Rzz block
			for i in range(n_spins):
				circuit.rx(p[count],i)
				count = count +1

			circuit.barrier()

			for i in range(n_spins-1):
				circuit.rzz(p[count],i,i+1)
				count = count +1

			circuit.barrier()

		if(j%2 == 1):
			for i in range(n_spins):
				circuit.ry(p[count],i)
				count = count +1

			circuit.barrier()

			for i in range(n_spins-1):
				circuit.rzz(p[count],i,i+1)
				count = count +1

			circuit.barrier()

	# Final block to close the ansatz
	if (depth%2 == 1):
		for i in range(n_spins):
				circuit.ry(p[count],i)
				count = count +1
	if (depth%2 == 0):
		for i in range(n_spins):
				circuit.rx(p[count],i)
				count = count +1

	return circuit

#==========================================

















