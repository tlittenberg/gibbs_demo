# gibbs_demo
Demonstration of Gibbs sampling

## Description

## Getting Started

### Dependencies
 + cmake
 + gsl

### Installing
`mpicc gibbs.c -lm -lgsl -o gibbs`

### Executing the program

Example Usage: `mpirun-openmpi-mp -np 3 gibbs`

### Output file format
The program outputs 5 different Markov chain files.

Three copies using the same sampler but different seeds and number of steps (to check internal consistency of samplers)
  + `standard_mcmc_0.dat`
  + `standard_mcmc_1.dat`
  + `standard_mcmc_2.dat`

And two different ways of Gibbs sampling.  One in serial (the right way) `serial_gibbs.dat` and one in parallel (the approximate way) `parallel_gibbs.dat`

The chain files all have the same format: Each row is a sample from the chain, and each column is on of the parameters:

     x[0] y[0] z[0]
     x[1] y[1] z[1]
     x[2] y[2] z[2]
     ...

An included `jupyter` notebook `GibbsTests.ipynb` displays corner plots of the posteriors.


## Authors
 + Tyson B. Littenberg

 ## License

This project is licensed under the GNU General Public License v3.0  - see the LICENSE.md file for details
