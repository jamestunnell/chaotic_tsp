"""Solve the traveling salesman problem (TSP) by chaotic simulated annealing 
(CSA), using a transiently chaotic neural network (TCNN).

Usage:
    csa_tsp.py TSP_FILE [options]

Arguments:
    TSP_FILE The TSPLIB XML file containing distances between cities (TSP problem instance).

Options:
    --nruns=NRUNS       Number of times to try CSA on the TSP instance [default: 1]
    --maxiter=MAXITER   Maximum iterations to run CSA for [default: 1000]
    --k=K               Damping factor of nerve membrane, between 0 and 1 [default: 0.9]
    --alpha=ALPHA       Positive scaling parameter for neuronal inputs [default: 0.015]
    --beta=BETA         Damping factor of self-connection weight, between 0 and 1 [default: 0.01]
    --z0=SELFCONN       Self-connection weight start value [default: 0.1]
    --I0=INPUTBIAS      Input bias [default: 0.5]
    --epsilon=EPSILON   Steepness parameter of neuron output function [default: 0.004]
    --W1=VALIDITYWT     Weight of validity constraint [default: 1]
    --W2=OPTIMALITYWT   Weight of tour optimality constraint [default: 1]
    --energy            Collect energy data and add to graph as line plot
    --percent           Collect percent valid data and add to graph as line plot
    --length            Collect tour_length data and add to graph as histogram
"""

from docopt import docopt

args = docopt(__doc__,version="CSA TSP 1.0")
print("Running CSA TSP with args %s" % args)

import tcnn
import tsplib
import matplotlib.pyplot as plt
import numpy as np

N_RUNS = int(args["--nruns"])
MAX_IT = int(args["--maxiter"])
plot_energy = args["--energy"]
plot_percent_valid = args["--percent"]
plot_tour_length = args["--length"]
tsp_file = args["TSP_FILE"]
constants = {
    "k": float(args["--k"]),
    "epsilon": float(args["--epsilon"]),
    "I0": float(args["--I0"]),
    "z0": float(args["--z0"]),
    "W1": float(args["--W1"]),
    "W2": float(args["--W2"]),
    "alpha": float(args["--alpha"]),
    "beta": float(args["--beta"])
}

attrs = ["iter"]
n_plots = 0
if plot_energy:
    n_plots += 1
    attrs.append("energy")
    
if plot_tour_length:
    n_plots += 1
    
if plot_percent_valid:
    n_plots += 1
    attrs.append("percent_valid")

distances = tsplib.distance_matrix(tsp_file)

tour_lengths = []
for i in range(N_RUNS):
    net = tcnn.TCNN(distances, **constants)
    results = net.run(maxiter = MAX_IT, collecting = attrs)
    I = results["iter"]
    
    if net.valid_tour():
        l = net.tour_length()
        tour_lengths.append(l)
        print("run %d converged by step %d, length = %f" % (i, I[-1],l))
    else:
        print("run %d did not converge by step %d" % (i,I[-1]))
    
    cur_plot = 1
    if plot_percent_valid:
        plt.subplot(n_plots, 1, cur_plot)
        plt.plot(I, results["percent_valid"])
        plt.xlabel('Iteration')
        plt.ylabel('Percent valid')
        cur_plot += 1
    
    if plot_energy:
        plt.subplot(n_plots, 1, cur_plot)
        plt.plot(I, results["energy"])
        plt.xlabel('Iteration')
        plt.ylabel('Energy')
        cur_plot += 1

if plot_tour_length:
    nbins = np.floor(N_RUNS/np.log2(N_RUNS))
    plt.subplot(n_plots, 1, cur_plot)
    plt.hist(tour_lengths,bins=nbins)
    plt.xlabel('Tour Lengths')    
    cur_plot += 1

if n_plots > 0:
    plt.show()
