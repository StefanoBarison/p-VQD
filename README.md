# Projected - Variational Quantum Dynamics (p-VQD)

[![DOI](https://zenodo.org/badge/359467906.svg)](https://zenodo.org/badge/latestdoi/359467906)

This repository contains the source code of the p-VQD algorithm to reproduce the results of the paper "An efficient quantum algorithm for the time evolution of parameterized circuits" (https://arxiv.org/abs/2101.04579).

At the moment, the Qiskit version required is


|qiskit                   | 0.25.0  |
|-------------------------|---------|
|qiskit-aer               | 0.8.0   |
|qiskit-aqua              | 0.9.0   |
|qiskit-finance           | 0.1.0   |
|qiskit-ibmq-provider     | 0.12.2  |
|qiskit-ignis             | 0.6.0   |
|qiskit-machine-learning  | 0.1.0   |
|qiskit-nature            | 0.1.0   |
|qiskit-optimization      | 0.1.0   |
|qiskit-terra             | 0.17.0  |

## Content of the repository

- **pVQD.py** : contains the actual class of the pVQD algorithm , with methods to evaluate the overlap, the gradient, the observables and run 
- **pauli_function.py** : contains functions to prepare Pauli operators and Hamiltonians for different physical systems. In this case, we create the Transverse Field Ising Model
- **ansatze.py** : a file to contain the functions that return quantum circuit ansatze, in the future it will be updated with new ansatze
- **example.py** : an example to use the pVQD to simulate the Transverse Field Ising Model on an open chain of 3 qubits
- **plotter.py** : simple script to plot the results of the pVQD calculations and compare them to the exact classical simulation
- **data** : a folder that contains some pre-produced data to plot



## Updates

- **Update 1** : the code has been updated to run with Qiskit 0.25.0, and uses new functions that have been introduced with it, like the PauliSumOp operator. Moreover, it creates the circuit only once in order to speed up calculations. Finally, the local cost function alternative proposed by Cerezo et al. (https://www.nature.com/articles/s41467-021-21728-w) has been introduced as a possibility.