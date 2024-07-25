import pytest
import numpy as np
from simulate_singlemolecules import simulate_single, steady_state

def test_simulate():
	'''
	Tries to simulate a trajectory and make a plot
	'''

	rates = np.array(([[0,3.],[8.,0.]]))
	emissions = np.array((0.,1.))
	noise = 0.05 
	nframes = 1000000
	dt = .01

	trajectory,signal = simulate_single(rates,emissions,noise,nframes,dt)

	q = rates.copy()
	p = np.zeros(q.shape[0])
	for i in range(q.shape[0]):
		q[i,i] = -q[i].sum()
		p[i] = (trajectory[1][trajectory[0] == i]).sum()
	p /= p.sum()

	p0 = steady_state(q)
	rel = np.abs(p[0]-p0[0])/p0[0]

	print('Steady State:',p0)
	print('Simulation  :',p)
	print('Relative:',rel)
	assert(rel < 0.01)

def test_numba():
	## There's a weird FNV warning these days...
	import numba as nb
	@nb.njit
	def fake():
		return True
	assert fake()
