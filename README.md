# Projected - Variational Quantum Dynamics (p-VQD)

This repository contains the source code of the p-VQD algorithm to reproduce the results of the paper "An efficient quantum algorithm for the time evolution of parameterized circuits" (https://arxiv.org/abs/2101.04579).

At the moment, the Qiskit version required is


| qiskit              | 0.24.0 |
|---------------------|--------|
| qiskit-aer          | 0.7.6  |
| qiskit-aqua         | 0.8.2  |
| qiskit-ibm-provider | 0.12.1 |
| qiskit-ignis        | 0.5.2  |
| qiskit-terra        | 0.16.4 |



## Content of the repository

- **pVQD.py** : contains the actual class of the pVQD algorithm , with methods to evaluate the overlap, the gradient, the observables and run 
- **create_hamiltonians.py** : contains functions to prepare Pauli operators and Hamiltonians for different physical systems (e.g. Ising model, Heisenberg model, ...)
- **ansatze.py** : a file to contain the functions that return quantum circuit ansatze, in the future it will be updated with new ansatze
- **example.py** : an example to use the pVQD to simulate the Transverse Field Ising Model on an open chain of 3 qubits