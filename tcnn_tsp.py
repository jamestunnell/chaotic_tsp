import numpy as np
import math

class TCNNTSP:
    """
    Implements a Transiently Chaotic Neural Network (TCNN), as presented in
    "Chaotic Simulated Annealing by a Neural Network Model with Transient Chaos"
    by Chen and Aihara, 1995.
    
    This TCNN is specifically tailored for solving the traveling salesman 
    problem (TSP). For a n-city TSP, there will be a total of N = n^2 neurons,
    or n rows of n neurons.
    """
    def __init__(self, distances, **constants):
        m,n = distances.shape
        if m != n:
            raise RuntimeError("Distance matrix is not square")
        
        self.real_distances = distances
        self.norm_distances = distances / distances.max()
        self.n = n
        ns = range(n)
        self.ns = ns
        
        self.W1 = constants["W1"]
        self.W2 = constants["W2"]
        self.alpha = constants["alpha"]
        self.beta = constants["beta"]
        self.epsilon = constants["epsilon"]
        self.k = constants["k"]
        self.z0 = constants["z0"]
        self.I0 = constants["I0"]

        self.z = self.z0        
        self.X = np.zeros((n,n))
        self.Y = np.zeros((n,n))

        self.X += np.random.uniform(-1, 1, (n,n))
        self.pairs = self.__random_pairs()
        self.iter = 0

    def __random_pairs(self):
        pairs = []
        for i in self.ns:
            for j in self.ns:
                pairs.append((i,j))
                
        np.random.shuffle(pairs)
        return pairs
            
    def __g(self,x):
        #return 1.0 / (1 + math.exp(-x / self.epsilon))
        return 0.5 * (1 + math.tanh(x/self.epsilon))

    def __retrieve(self, attr):
        res = getattr(self,attr)
        if callable(res):
            return res()
        else:
            return res
        
    def run(self, maxiter=None, collecting=None):
        results = {"steps":[]}
        if collecting:
            for attr in collecting:
                results[attr] = []
        
        iters = 0
        while not self.valid_tour() and (iters < maxiter if maxiter else True):
            self.step()
            if collecting:
                for attr in collecting:
                    results[attr].append(self.__retrieve(attr))
            iters += 1
        return results
    
    def step(self):
        # update each neuron asynchronously at random
        #for i,k in self.__random_pairs():
        for i,k in self.pairs:
            self.__update_neuron(i,k)
        
        self.z *= (1 - self.beta)
        self.iter += 1

    def __update_output(self,i,k):
        self.Y[i,k] = self.__g(self.X[i,k])
        
    # update neuron internal state, then output
    def __update_neuron(self,i,k):
        n, X, Y = self.n, self.X, self.Y
        W1, W2, alpha = self.W1, self.W2, self.alpha
        ns, ds = self.ns, self.norm_distances
        
        a = -W1*(
            sum(Y[i,l] if l != k else 0.0 for l in ns) + 
            sum(Y[j,k] if j != i else 0.0 for j in ns)
        )
        b = -W2*sum(ds[i,j]*(Y[j,(k+1)%n] + Y[j,(k-1)%n]) if j != i else 0.0 for j in ns)
        
        c = self.k*X[i,k] - self.z*(Y[i, k] - self.I0)
        
        X[i,k] = alpha*(a + b + W1) + c
        self.__update_output(i,k)
        
    def energy(self):
        W1, W2, Y = self.W1, self.W2, self.Y
        n, ns, ds = self.n, self.ns, self.norm_distances
        
        temp1 = sum((sum(Y[i,k] for k in ns) - 1.0)**2 for i in ns)
        temp2 = sum((sum(Y[i,k] for i in ns) - 1.0)**2 for k in ns)
        
        temp3 = 0.0
        for i in ns:
            for j in ns:
                for k in ns:
                    temp3 += ds[i,j]*Y[i,k]*(Y[j,(k+1)%n] + Y[j,(k-1)%n])
        
        E1 = 0.5*W1*(temp1 + temp2)
        E2 = 0.5*W2*temp3
        return E1 + E2
        
    def valid_rows(self):
        return [ len(np.where(self.Y[i])[0]) == 1 for i in self.ns ]
    
    def valid_cols(self):
        YT = self.Y.transpose()
        return [ len(np.where(YT[i])[0]) == 1 for i in self.ns ]
        
    def n_valid_rows(self):
        return len(np.where(self.valid_rows())[0])
        
    def n_valid_cols(self):
        return len(np.where(self.valid_cols())[0])
    
    def percent_valid(self):
        total = self.n_valid_rows() + self.n_valid_cols()
        return total / (2.0 * self.n)

    def valid_tour(self):
        return self.percent_valid() == 1
    
    def tour(self):
        if not self.valid_tour:
            raise RuntimeError("Tour is not valid!")
        
        ns, YT = self.ns, self.Y.transpose()
        return [ np.where(YT[i])[0][0] for i in ns ]
        
    def tour_length(self):
        n, ns, ds = self.n, self.ns, self.real_distances
        
        tour = self.tour()
        citypairs = [(tour[i], tour[(i+1)%n]) for i in ns ]
        distances = [ ds[cp[0],cp[1]] for cp in citypairs ]
        return sum(distances)