import numpy as np
import tsplib
import matplotlib.pyplot as plt

def tour_length(tour, distances):
    ds = distances
    m,n = distances.shape
    
    if m != n:
        raise ValueError("The distance matrix is not square")
        
    if len(tour) != n:
        raise ValueError("The tour length does not match the distance matrix size")
    
    ns = range(n)
    citypairs = [(tour[i], tour[(i+1)%n]) for i in ns ]
    distances = [ ds[cp[0],cp[1]] for cp in citypairs ]
    return sum(distances)


tsp_file = "tsp_files/burma14.xml"
#tsp_file = "tsp_files/berlin52.xml"

distances = tsplib.distance_matrix(tsp_file)
m,n = distances.shape

lens = []
best_tour = None

for i in range(1000):
    tour = range(n)
    np.random.shuffle(tour)
    l = tour_length(tour, distances)
    lens.append(l)
    #print(l)

    if best_tour is None:
        best_tour = (tour, l)
        print(best_tour)
    else:
        if l < best_tour[1]:
            t = tour
            best_tour = (t, l)
            print(best_tour)

plt.hist(lens, bins=20)
plt.show()