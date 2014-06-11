chaotic_tsp
===========

Perform chaotic simulated annealing (CSA), using transiently chaotic neural network (TCNN) to solve the traveling salesman problem (TSP).

## Background

Read my [blog post](http://jamestunnell.github.io/projects/2014/06/09/chaotic-tsp/) for an explanation of how CSA is used to solve the TSP.

## Installation

Just download and put it somewhere to run via command line. But before running anything, there are several dependencies to take care of: numpy, BeatifulSoup, matplotlib, and docopt. Install those via `pip`, `easy_install`, or whatever you prefer.

## Usage

The core functionality is in `tcnn.py`, but it's really more convenient to run the command line file `csa_tsp.py`, so we'll talk about that instead.

The file does have a python shebang, so in Linux you can first make it an executable:

<pre><code>sudo chmod +x csa_tsp.py</code></pre>

But before running the thing, take a look at how to use it:

<pre><code>$ ./csa_tsp.py --help</code></pre>

As you can see there are many options. Don't worry, the defaults will work fine for small TSP instances (for instance, the 14-city TSP, burma14). We'll give it a try, adding the `nruns` option to run CSA `nruns` times.

<pre><code>./csa_tsp.py tsp_files/burma14.xml --nruns 4</code></pre>

The output will be something like what is shown below. One of the solutions found was at the global minimum, 3323. 

<pre><code>
Running CSA TSP with args {'--I0': '0.5',
 '--W1': '1',
 '--W2': '1',
 '--alpha': '0.015',
 '--beta': '0.01',
 '--energy': False,
 '--epsilon': '0.004',
 '--k': '0.9',
 '--length': False,
 '--maxiter': '1000',
 '--nruns': '4',
 '--percent': False,
 '--z0': '0.1',
 'TSP_FILE': 'tsp_files/burma14.xml'}
run 0 converged by step 209, length = 3369.000000
run 1 converged by step 209, length = 3346.000000
run 2 converged by step 207, length = 3323.000000
run 3 converged by step 209, length = 3409.000000
</code></pre>

