import numpy as np  
import functools
import itertools
import matplotlib.pyplot as plt 
from scipy   import  linalg as LA 
import json

from qiskit import IBMQ, Aer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

from qiskit.aqua                      import QuantumInstance
from qiskit.aqua.operators            import Z, I, X, Y
from qiskit.aqua.operators 			  import PauliOp, SummedOp

from pauli_function import *
from pVQD			import *
from ansatze        import *

# Initialize system parameters for Ising

spins   = 3
V       = 0.25
g       = 1.0
dt      = 0.05
n_steps = 40

# Algorithm parameters

ths = 0.99999
depth = 3


### Example circ

ex_params = np.zeros((depth+1)*spins +depth*(spins-1))
wfn = hweff_ansatz(ex_params)


### Shift
shift  = np.array(len(ex_params)*[0.01])

print("Initial shift:",shift)


### Generate the Hamiltonian
H = generate_ising(spins,V,g)

print(wfn)
print(H)

### Backend
shots = 800
backend  = Aer.get_backend('qasm_simulator')
instance = QuantumInstance(backend=backend,shots=shots)

### Prepare the observables to measure
obs = {}
# Magnetization

for i in range(spins):
	obs['Sz_'+str(i)]      = PauliOp(generate_pauli([],[i],spins),1.0) 
	obs['Sx_'+str(i)]      = PauliOp(generate_pauli([i],[],spins),1.0)
	obs['Sy_'+str(i)]      = PauliOp(generate_pauli([i],[i],spins),1.0)


for (name,pauli) in obs.items():
	print(name)
	print(pauli)


### Initialize the algorithm

# Choose a specific set of parameters
initial_point = None

# Choose the gradient optimizer: 'sgd', 'adam'
gradient = 'sgd'


algo = pVQD(H,hweff_ansatz,ex_params,shift,instance,shots)
algo.run(ths,dt,n_steps, obs_dict = obs,filename= 'data/trial_results.dat', max_iter = 50)

