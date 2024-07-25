from setuptools import setup, find_packages

setup(
	name='simulate_singlemolecules',
	version='0.1',
	packages=find_packages(),
	install_requires=[
		'numpy',
		'numba',
	],
	extras_require={
		'test': [
			'pytest',
			'pytest-cov',
		],
	},
	entry_points={
		'console_scripts': [
			# Define any CLI commands here
		],
	},
)