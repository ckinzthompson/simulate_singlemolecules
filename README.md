# simulate_singlemolecules
Simulate single molecule trajectories

## Install
``` sh
git clone https://github.com/ckinzthompson/biasd.git
cd biasd
pip install
```

## Functions
* `steady_state(Q)`: Calculates the steady state probabilities from a Q matrix.
* `simulate_single(rates, emissions, noise, nframes, dt)`: Simulates a single trajectory with given rates, emissions, and noise.
* `simulate_ensemble(rates, emissions, noise, nframes, dt, nmol)`: Simulates an ensemble of trajectories.
* `simulate_fret(rates, emissions, noise, nframes, dt, nmol, nphotons)`: Simulates FRET signals for an ensemble of molecules.

## Testing/Development
``` sh
pip install -e ".[test]"
pytest
```