from tcnn_tsp import TCNNTSP
import tsplib
import matplotlib.pyplot as plt

tsp_file = "tsp_files/burma14.xml"
#tsp_file = "tsp_files/berlin52.xml"
distances = tsplib.distance_matrix(tsp_file)

valid_tours = []
best_tour = None

constants = {
    "k": 0.9, "epsilon": 1/250., "I0": 0.5, "z0": 0.1, "W1": 1, "W2": 1.,
    "alpha": 0.015, "beta": 5e-3
}

N_RUNS = 50
GLOBAL_MIN = 3323.0
MAX_ITER = 2000

errors = []
for i in range(N_RUNS):
    net = TCNNTSP(distances, **constants)
    steps = 0
    for energy, percent_valid in net.run(MAX_ITER):
        steps += 1
    
    if net.valid_tour():
        l = net.tour_length()
        errors.append(abs(GLOBAL_MIN - l)/GLOBAL_MIN)
        print("run %d converged after %d steps" % (i,steps))

import matplotlib.pyplot as plt
plt.hist(errors)
plt.show()
