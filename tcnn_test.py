from tcnn_tsp import TCNNTSP
import tsplib
import matplotlib.pyplot as plt
import numpy as np

tsp_file = "tsp_files/burma14.xml"
#tsp_file = "tsp_files/berlin52.xml"
distances = tsplib.distance_matrix(tsp_file)

valid_tours = []
best_tour = None

constants = {
    "k": 0.9, "epsilon": 1/250., "I0": 0.5, "z0": 0.1, "W1": 1, "W2": 1.,
    "alpha": 0.015, "beta": 1e-2
}

N_RUNS = 20
MIN_RUNS_FOR_HIST = 20
GLOBAL_MIN = 3323.0
MAX_ITER = 500

n_plots = 2
if N_RUNS >= MIN_RUNS_FOR_HIST:
    n_plots = 3

lens = []
for i in range(N_RUNS):
    net = TCNNTSP(distances, **constants)
    
    attrs = ["iter","energy","percent_valid"]
    results = net.run(maxiter = MAX_ITER, collecting = attrs)
    I = results["iter"]
    
    if net.valid_tour():
        l = net.tour_length()
        lens.append(l)
        print("run %d converged by step %d, length = %f" % (i, I[-1],l))
    else:
        print("run %d did not converge by step %d" % I[-1])
        
    plt.subplot(n_plots, 1, 1)
    plt.plot(I, results["percent_valid"])
    plt.xlabel('Iteration')
    plt.ylabel('Percent valid')
    
    plt.subplot(n_plots, 1, 2)
    plt.plot(I, results["energy"])
    plt.xlabel('Iteration')
    plt.ylabel('Energy')

if N_RUNS >= MIN_RUNS_FOR_HIST:
    nbins = np.floor(N_RUNS/np.log2(N_RUNS))
    plt.subplot(n_plots, 1, 3)
    plt.xlabel('Tour Lengths')
    plt.hist(lens,bins=nbins)

plt.show()
