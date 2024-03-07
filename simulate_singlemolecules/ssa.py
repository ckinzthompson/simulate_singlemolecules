import numpy as np
import numba as nb

################################################################################
######### SSA
################################################################################

def steady_state(Q):
	'''
	Calculates the steady state probabilities the states from a Q matrix
	'''
	
	## Calculate the Eigenvalues and Eigenvectors of Q^T
	D,P = np.linalg.eig(Q.T)
	
	## Get the eigenvector with eigenvalue of 0.0 (if a TM, it'd be 1...)
	eigenval_index = np.where(D==0.0)[0][0]
	P_ss = P[:,eigenval_index]

	## Normalize the eigenvector from a basis vector into a probability
	P_ss /= P_ss.sum()

	return P_ss


def gen_dwells(rates,tlength,seed):
	'''
	Input:
		* `rates` is a KxK np.ndarray of the rate constants for transitions between states i and j. It should have zeros on the diagonals
		* `tlength` is a float/double that specifies the total time of the trajectory
	Output
		* a 2xN np.ndarray of states indices (0) and the corresponding dwell times (1)
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
	traj = ssa(tlength,initial_state,rates)
	return traj

@nb.njit
def ssa(tlength,initial_state,rates):

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

################################################################################
######### Signal
################################################################################

@nb.njit
def render_trajectory(trajectory,steps,dt,emission):
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
######### Simulate
################################################################################

def simulate_single(rates,emissions,noise,nframes,dt):
	'''
	Input:
		* `rates` is a KxK np.ndarray of the rate constants for transitions between states i and j. It should have zeros on the diagonals
		* `emissions` is a np.ndarray of length K with the emission means of each state
		* `noise` is a float with the standard deviation of the normal distribution used to add noise to the signal
		* `nframes` is an integer number of datapoints in the signal versus time trajectory
		* `dt` is the time period of each datapoint
	Output:
		* `trajectory` is a 2xN np.ndarray of states indices (0) and the corresponding dwell times (1) of the state trajectory
		* `signal` is a 2x`nframes` np.ndarray containing the time points (0) and the signal values (1) of the rendered signal trajectory
	'''

	np.random.seed()
	trajectory = gen_dwells(rates,nframes*dt)
	signal = render_trajectory(trajectory,nframes,dt,emissions)
	signal[1] += np.random.normal(size=signal.shape[1])*noise
	return trajectory,signal

def simulate_ensemble(rates,emissions,noise,nframes,dt,nmol):
	out = np.zeros((nmol,nframes))
	for i in range(nmol):
		trajectory,signal = simulate_single(rates,emissions,noise,nframes,dt)
		out[i] = signal[1]
	return out

def simulate_fret(rates,emissions,noise,nframes,dt,nmol):
	out = np.zeros((nmol*2,nframes))
	for i in range(nmol):
		trajectory,signal = simulate_single(rates,emissions,noise,nframes,dt)
		out[i*2+1] = 5000.*signal[1]
		out[i*2] = 5000.*(1.-signal[1])
	return out.T

def test():
	'''
	Tries to simulate a trajectory and make a plot
	'''

	import matplotlib.pyplot as plt

	# rates = np.array(([0,10.,2.],[2.,0,2.],[1.5,2.,0]))
	# emissions = np.array((0.,1.,2.))
	rates = np.array(([[0,3.],[8.,0.]]))
	emissions = np.array((0.,1.))
	noise = 0.05 # SNR = 20
	nframes = 1000000
	dt = .001 # 500 msec

	trajectory,signal = simulate_single(rates,emissions,noise,nframes,dt)

	q = rates.copy()
	p = np.zeros(q.shape[0])
	for i in range(q.shape[0]):
		q[i,i] = -q[i].sum()
		p[i] = (trajectory[1][trajectory[0] == i]).sum()
	p /= p.sum()

	print('Steady State:',steady_state(q))
	print('Simulation  :',p)

	stop = np.min((nframes,2000))

	tt = np.zeros(trajectory.shape[1]*2)
	tt[::2] = trajectory[1].cumsum()
	tt[1::2] = trajectory[1].cumsum()
	yy = np.zeros(trajectory.shape[1]*2)
	yy[::2] = trajectory[0]
	yy[1::2] = np.roll(trajectory[0],-1)

	plt.plot(tt[:-1],yy[:-1],alpha=.8)
	plt.plot(signal[0,:stop],signal[1,:stop],'o',alpha=.5)
	plt.xlim(0,signal[0,stop-1])

	plt.xlabel('Time',fontsize=12)
	plt.ylabel('Signal',fontsize=12)
	plt.title('Blurred SSA Trajectory',fontsize=16)
	plt.show()

if __name__ == '__main__':
	test()
