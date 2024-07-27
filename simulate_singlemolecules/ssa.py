import numpy as np
import numba as nb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="FNV hashing is not implemented in Numba")

def _gen_dwells(rates,tlength):
	'''
	Generates dwell times for a given rate matrix and total time length.
	
	Parameters:
	rates (np.ndarray): A KxK array of rate constants for transitions between states. It should have zeros on the diagonals
	tlength (float): Total time length of the trajectory.
	
	Returns:
	np.ndarray: A 2xN array of state indices (0) and corresponding dwell times (1).
	'''
	np.random.seed()

	## create Q matrix from K matrix
	q = rates.copy()
	for i in range(q.shape[0]):
		q[i,i] = - q[i].sum()

	pst = steady_state(q)

	## find intial state
	p = np.random.rand()
	initial_state = np.searchsorted(pst.cumsum(),p)
	
	## simulate trajectory
	traj = _ssa(tlength,initial_state,rates)
	return traj

@nb.njit
def _ssa(tlength,initial_state,rates):
	'''
	Stochastic simulation algorithm for generating state trajectories.

	Parameters:
	tlength (float): Total time of the trajectory.
	initial_state (int): Initial state index.
	rates (np.ndarray): A KxK array of rate constants for transitions between states. It should have zeros on the diagonals

	Returns:
	np.ndarray: A 2xN array of state indices (0) and corresponding dwell times (1).
	'''

	nstates = rates.shape[0]
	steps = 2*int(np.floor(tlength*np.max(rates)))
	if steps < 10:
		steps = 10

	states_dwells = np.zeros((2,steps), dtype=np.double)

	currentstate = initial_state
	timetotal = 0.
	cumulatives = np.zeros_like(rates)
	outrates = np.zeros(nstates)

	## initialize
	for i in range(nstates):
		for j in range(nstates):
			cumulatives[i,j] = rates[i,j]
			if j > 0:
				cumulatives[i,j] += cumulatives[i,j-1]
		outrates[i] = cumulatives[i,-1]

	## simulate
	for i in range(steps):
		## get dwelltime
		states_dwells[0,i] = currentstate
		states_dwells[1,i] = np.random.exponential(1./outrates[int(states_dwells[0,i])])

		## start the trace at a random time
		if i == 0:
			states_dwells[1,i] *= np.random.rand()
		timetotal += states_dwells[1,i]

		## pick new state
		r = np.random.rand()
		for j in range(nstates):
			currentstate = j
			if cumulatives[int(states_dwells[0,i])][j] >= r*outrates[int(states_dwells[0,i])]:
				break

		if timetotal > tlength:
			break

	# if np.size(states) == 0:
	# 	states = [initialstate]
	# 	dwells = [tlength]
	return states_dwells[:,:i+1]

@nb.njit
def _render_trajectory(trajectory,steps,dt,emission):
	'''
	Input:
		* `trajectory` is the 2xM output from the `ssa_dwells` function
		* `steps` is the integer number of discrete timepoints to render
		* `dt` is a float specifying the period of each timepoint
		* `emission` is a np.ndarray of length K with the emission means of each state
	Output:
		* a np.ndarray of shape 2x`steps` containing the time points (0) and the signal values (1) of the rendered trajectory
	'''

	nstates = emission.size
	timesteps = trajectory.shape[1]

	times = np.zeros(timesteps+1,dtype=nb.double)
	for i in range(timesteps):
		times[i+1] = trajectory[1,i] + times[i]

	a = 0
	b = 0

	out = np.zeros((2,steps))

	for i in range(steps):
		out[0,i] = (i+1)*dt
		t1 = out[0,i]
		t0 = t1 - dt
		aflag = True
		bflag = True

		for j in range(a,timesteps+1):
			if (times[j] > t0) and aflag:
				a = j-1
				aflag = False
			if (times[j] > t1) and bflag:
				b = j
				bflag = False
			if (not aflag) and (not bflag):
				out[1,i] = 0.
				if (b-a) == 1:
					out[1,i] += (t1-t0)/dt * emission[int(trajectory[0,a])]
				elif (b-a) > 1:
					out[1,i] += (times[a+1]-t0)/dt * emission[int(trajectory[0,a])]
					for j in range(a+1,b-1):
						out[1,i] += trajectory[1,j]/dt * emission[int(trajectory[0,j])]
					out[1,i]+= (t1 - times[b-1])/dt * emission[int(trajectory[0,b-1])]
				break
	return out

################################################################################
######### Callable Functions
################################################################################


def steady_state(Q):
	'''
	Calculates the steady state probabilities the states from a Q matrix

	Parameters:
	Q (np.ndarray): A square matrix of transition rates between states.

	Returns:
	np.ndarray: Steady state probabilities of each state.
	'''
	
	## Calculate the Eigenvalues and Eigenvectors of Q^T
	D,P = np.linalg.eig(Q.T)
	
	## Get the eigenvector with eigenvalue of 0.0 (if a TM, it'd be 1...)
	## Note, numerical issues mean you can get O(10^-16) values. Use 10 fold as a heuristic padding.
	cutoff = np.finfo(np.float64).eps*10.0
	eigenval_index = np.where(np.abs(D)<cutoff)[0][0]
	P_ss = P[:,eigenval_index]

	## Normalize the eigenvector from a basis vector into a probability
	P_ss /= P_ss.sum()

	return P_ss



def simulate_single(rates,emissions,noise,nframes,dt):
	'''
	Simulates a single trajectory with given rates, emissions, and noise.

	Parameters:
	rates (np.ndarray): A KxK array of rate constants for transitions between states. It should have zeros on the diagonals
	emissions (np.ndarray): Emission means for each state. K-sized array
	noise (float): Standard deviation of the normal distribution used to add noise. K-sized array
	nframes (int): Number of data points in the signal vs. time trajectory.
	dt (float): Time period of each data point.
	'''

	np.random.seed()
	trajectory = _gen_dwells(rates,nframes*dt)
	signal = _render_trajectory(trajectory,nframes,dt,emissions)
	signal[1] += np.random.normal(size=signal.shape[1])*noise
	return trajectory,signal

def simulate_ensemble(rates,emissions,noise,nframes,dt,nmol):
	'''
	Simulates an ensemble of trajectories.

	Parameters:
	rates (np.ndarray): A KxK array of rate constants for transitions between states. It should have zeros on the diagonals
	emissions (np.ndarray): Emission means for each state. K-sized array
	noise (float): Standard deviation of the normal distribution used to add noise. K-sized array
	nframes (int): Number of data points in the signal vs. time trajectory.
	dt (float): Time period of each data point.
	nmol (int): Number of molecules to simulate.

	Returns:
	trajectories (np.ndarray): An nmol x nframes array of simulated signal values.
	'''

	out = np.zeros((nmol,nframes))
	for i in range(nmol):
		trajectory,signal = simulate_single(rates,emissions,noise,nframes,dt)
		out[i] = signal[1]
	return out

def simulate_fret(rates,emissions,noise,nframes,dt,nmol,nphotons=5000.):
	'''
	Simulates FRET signals for an ensemble of molecules.

	Parameters:
	rates (np.ndarray): A KxK array of rate constants for transitions between states. It should have zeros on the diagonals
	emissions (np.ndarray): Emission means for each state. K-sized array
	noise (float): Standard deviation of the normal distribution used to add noise. K-sized array
	nframes (int): Number of data points in the signal vs. time trajectory.
	dt (float): Time period of each data point.
	nmol (int): Number of molecules to simulate.

	Returns:
	fret (np.ndarray): A (nmol,2,nframes) array of simulated FRET signal values.
	'''
	
	out = np.zeros((nmol,2,nframes))
	for i in range(nmol):
		trajectory,signal = simulate_single(rates,emissions,noise,nframes,dt)
		out[i,0] = nphotons*signal[1]
		out[i,1] = nphotons*(1.-signal[1])
	return out.T

def testdata(nmol=10,nt=1000):
	return simulate_ensemble(
		np.array(((0.,3.),(8.,0))),
		np.array((0.,1.)),
		0.05,
		nt,
		0.1,
		nmol
	)