import numpy as np 
import json
import functools
import itertools
import matplotlib.pyplot as plt 
from scipy   import  linalg as LA 



import qiskit
from qiskit.quantum_info 			     import Pauli
from qiskit.aqua.operators 			     import PauliOp, SummedOp
from qiskit.aqua.operators.evolutions    import Trotter, PauliTrotterEvolution

from qiskit.aqua.operators.state_fns     import CircuitStateFn
from qiskit.aqua.operators.expectations  import PauliExpectation, AerPauliExpectation, MatrixExpectation
from qiskit.aqua.operators.primitive_ops import CircuitOp
from qiskit.aqua.operators            	 import Z, I


from qiskit 							 import Aer, execute
from qiskit.aqua 						 import QuantumInstance
from qiskit.aqua.operators 				 import CircuitSampler, StateFn

from pauli_function import *


# This class aims to simulate the dynamics of a quantum system
# approximating it with a variational ansatz whose parameters 
# are varied in order to follow the unitary evolution

# Useful functions
def projector_zero(n_qubits):
	from qiskit.aqua.operators            import Z, I

	prj = (1/np.power(2,n_qubits))*(I+Z)

	for a in range(n_qubits-1):
		prj = prj^(I+Z)

	return prj


def projector_zero_local(k,n_qubits):
	from qiskit.aqua.operators            import Z, I

	

	prj_list = [I for i in range(n_qubits)]
	prj_list[k] = 0.5*(I+Z)

	prj = prj_list[0]

	for a in range(1,len(prj_list)):
		prj = prj^prj_list[a]

	return prj

def ei(i,n):
	vi = np.zeros(n)
	vi[i] = 1.0
	return vi[:]



class pVQD:

	def __init__(self,hamiltonian,ansatz,parameters,initial_shift,instance,shots):

		'''
		Args:
		
		parameters    : [numpy.array] an array containing the parameters of the ansatz
		initial_shift : [numpy.array] an array containing the initial guess of shifts

		'''
		self.hamiltonian     = hamiltonian
		self.ansatz          = ansatz
		self.instance        = instance 
		self.parameters      = parameters
		self.num_parameters  = len(parameters)
		self.shift           = initial_shift
		self.shots           = shots



	# This function calculate overlap and gradient of the overlap using a global operator on the |0><0| state

	def measure_overlap_and_gradient(self,time_step):
		# The aim of this function is to measure  
		# the quantity |<psi(w+dw)|U(dt)|psi(w)>|^2 and 
		# its gradient wrt dw
		
		# The operator to be measured here is |0><0|
		prj_zero = StateFn(projector_zero(self.hamiltonian.num_qubits),is_measurement = True)

		# Now let's create the Trotter operator 

		step_h = time_step*self.hamiltonian

		trotter = PauliTrotterEvolution(reps=1)
		U_dt    =trotter.convert(step_h.exp_i()).to_circuit()
		# let's construct the complete circuit V^\dag(w+dw)U(dt)V(w)

		# Measure the overlap and its gradient 
		nparameters  = len(self.parameters)
		shift_parameters = self.parameters + self.shift 

		wfn_circuits = [CircuitStateFn(self.ansatz(self.parameters)+U_dt+self.ansatz(shift_parameters).inverse())]


		for i in range(nparameters):
			wfn_circuits.append(CircuitStateFn(self.ansatz(self.parameters)+U_dt+self.ansatz(shift_parameters+ei(i,nparameters)*np.pi/2.0).inverse()))
			wfn_circuits.append(CircuitStateFn(self.ansatz(self.parameters)+U_dt+self.ansatz(shift_parameters-ei(i,nparameters)*np.pi/2.0).inverse()))

		# Now measure circuits
		results = []

		for wfn in wfn_circuits:
			braket     = prj_zero @ wfn
			grouped    = PauliExpectation().convert(braket)


			sampled_op = CircuitSampler(self.instance).convert(grouped)
			mean_value = sampled_op.eval().real
			est_err = 0

			if (not self.instance.is_statevector):
				variance = PauliExpectation().compute_variance(sampled_op).real
				est_err  = np.sqrt(variance/self.shots)

			results.append([mean_value,est_err])

		E = np.zeros(2)
		g = np.zeros((nparameters,2))

		E[0],E[1] = results[0]

		for i in range(nparameters):
			rplus  = results[1+2*i]
			rminus = results[2+2*i]
			# G      = (Ep - Em)/2
			# var(G) = var(Ep) * (dG/dEp)**2 + var(Em) * (dG/dEm)**2
			g[i,:] = (rplus[0]-rminus[0])/2.0,np.sqrt(rplus[1]**2+rminus[1]**2)/2.0

		self.overlap  = E
		self.gradient = g

		return E,g 



	def measure_aux_ops(self,pauli):

		wfn = CircuitStateFn(self.ansatz(self.parameters))
		op = StateFn(pauli,is_measurement = True)

		# Evaluate the aux operator given
		braket = op @ wfn
		grouped    = PauliExpectation(group_paulis=True).convert(braket)


		sampled_op = CircuitSampler(self.instance).convert(grouped)
		mean_value = sampled_op.eval().real
		est_err = 0

		if (not self.instance.is_statevector):
			variance = PauliExpectation().compute_variance(sampled_op).real
			est_err  = np.sqrt(variance/self.shots)

		res = [mean_value,est_err]

		return res

	def adam_gradient(self,count,m,v,g):
		## This function implements adam optimizer
		beta1 = 0.9
		beta2 = 0.999
		eps   = 1e-8
		alpha = [0.001 for i in range(len(self.parameters))]
		if count == 0:
			count = 1

		new_shift = [0 for i in range(len(self.parameters))]

		for i in range(len(self.parameters)):
			m[i] = beta1 * m[i] + (1 - beta1) * g[i]
			v[i] = beta2 * v[i] + (1 - beta2) * np.power(g[i],2)

			alpha[i] = alpha[i] * np.sqrt(1 - np.power(beta2,count)) / (1 - np.power(beta1,count))

			new_shift[i] = self.shift[i] + alpha[i]*(m[i]/(np.sqrt(v[i])+eps))

		return new_shift



	def run(self,ths,timestep,n_steps,obs_dict = None,filename='algo_result.dat',max_iter = 100,gradient='sgd',initial_point=None):


		times = [i*timestep for i in range(n_steps+1)]
		tot_steps= 0

		if initial_point != None :
			if len(initial_point) != len(self.parameters):
				print("TypeError: Initial parameters are not of the same size of circuit parameters")
				return

				
			print("\nRestart from: ")
			print(initial_point)
			self.parameters = initial_point

		print("Running the algorithm")

		# prepare contaners for observables
		if len(obs_dict) > 0:
			obs_measure = {}
			obs_error   = {}

			for (obs_name,obs_pauli) in obs_dict.items():
				first_measure                   = self.measure_aux_ops(obs_pauli)
				obs_measure[str(obs_name)]      = [first_measure[0]]
				obs_error['err_'+str(obs_name)] = [first_measure[1]]


		counter = []
		initial_fidelities = []
		fidelities = []
		err_fin_fid = []
		err_init_fid = []
		params = []

		params.append(list(self.parameters))

		for i in range(n_steps):


			print('\n================================== \n')
			print("Time slice:",i+1)
			print("Shift before optimizing this step:",self.shift)
			print("Initial parameters:", self.parameters)
			print('\n================================== \n')
			

			count = 0
			self.overlap = [0.01,0]
			g_norm = 1

			if gradient == 'adam':
				m = np.zeros(len(self.parameters))
				v = np.zeros(len(self.parameters))
			
			# Optimize the shift

			while self.overlap[0] < ths and count < max_iter: 
				print("Shift optimizing step:",count+1)
				count = count +1 
				E,g = self.measure_overlap_and_gradient(timestep)

				tot_steps= tot_steps+1

				if count == 1:
					initial_fidelities.append(self.overlap[0])
					err_init_fid.append(self.overlap[1])

				print('Overlap',self.overlap)
				print('Gradient',self.gradient[:,0])

				if gradient == 'adam':
					print("\n Adam \n")
					grad = np.asarray(g[:,0])
					self.shift = np.asarray(self.adam_gradient(count,m,v,grad))
				else:
					self.shift = self.shift + g[:,0]

				#Norm of the gradient
				g_vec = np.asarray(g[:,0])
				g_norm = np.linalg.norm(g_vec)
				print('Gradient norm:',g_norm)

			# Update parameters

			print('\n---------------------------------- \n')

			print("Shift after optimizing:",self.shift)
			print("New parameters:"        ,self.parameters + self.shift)

			print("New overlap: "          ,self.overlap[0])

			self.parameters = self.parameters + self.shift


			# Measure quantities and save them 
			
			if len(obs_dict) > 0:

				for (obs_name,obs_pauli) in obs_dict.items():
					run_measure   = self.measure_aux_ops(obs_pauli)
					obs_measure[str(obs_name)].append(run_measure[0])
					obs_error['err_'+str(obs_name)].append(run_measure[1])


			counter.append(count)
			fidelities.append(self.overlap[0])
			err_fin_fid.append(self.overlap[1])
			
			params.append(list(self.parameters))


		# End of the algorithm

		print("Total measurements:",tot_steps)
		print("Measure per step:", tot_steps/n_steps)

		# Save data on file

		# Prepare a dictionary with the data
		log_data = {}
		if len(obs_dict) > 0:
			for (obs_name,obs_pauli) in obs_dict.items():
				log_data[str(obs_name)]        = obs_measure[str(obs_name)]
				log_data['err_'+str(obs_name)] = obs_error['err_'+str(obs_name)]
		
		log_data['init_F']      = initial_fidelities
		log_data['final_F']     = fidelities
		log_data['err_init_F']  = err_init_fid
		log_data['err_fin_F']   = err_fin_fid
		log_data['iter_number'] = counter
		log_data['times']       = times
		log_data['params']      = list(params)
		log_data['tot_steps']   = [tot_steps]

		# Dump on file
		json.dump(log_data, open( filename,'w+'))







