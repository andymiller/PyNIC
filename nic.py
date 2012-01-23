import numpy as np
import scipy as sp
import pylab as plt
import random,eval

class NIC:

  def __init__(self, k=2):
    self.K = k  
    self.labels_ = []

  def fit(self, X, y=None):
    """clusterNIC(samples, K)
      Implementation of the clustering algo from 
      A Nonparametric Information Theoretic Clustering Algorithm
      Faivishevsky and Goldberger, ICML 2010
    """
    N = len(X)
    
    #apply data whitening by mulitplying the data by cov(x)^(-1/2)
    #cov = np.cov(X.T)
    #whitened = np.multiply(X,cov) 
    whitened = X

    #cache the pairwise distances in all of X
    self.dists_ = np.log(sp.spatial.distance.pdist(X))
    self.dists_ = sp.spatial.distance.squareform(self.dists_)
     
    #initialize random assignment
    np.random.seed(0)
    self.labels_ = np.random.random_integers(0,self.K-1,N)    

    #Calculate score:
    sNIC = self.score(whitened); 
    print "SCORE: ", sNIC

    #do until convergence:
    # Go over the points in a circular manner. 
    conCount = 0
    converged = False
    idx = 0
    while not converged:
      # For data point xi calculate scores of all possible reassignments of xi to different clusters.
      jOrig = self.labels_[idx]
      scores = [] 
      for j in range(self.K):
        self.labels_[idx] = j
        scores.append(self.score(whitened)) 
      # Update current assignment C by choosing label ci that leads to the minimal score.
      self.labels_[idx] = scores.index(min(scores))
      # if a label changed, then no convergence
      if self.labels_[idx] != jOrig:
        conCount = 0
      else:
        conCount += 1
      if conCount >= N:
        print "... converged."
        converged = True
        break
      
      #increment circular
      idx = (idx+1) % N      
     
  def predict(self, X):
    """Predict cluster center for new points X"""
    #for i, xi in enumerate(X):
    #  scores = []
    #  for j in range(self.K):
    #    self.labels_[idx]= j
    pass            
 
  def score(self, X):
    """Caculates S_nic(cluster_centers_)"""
    score = 0.0
    for j in range(self.K):
      idx = self.labels_ == j
      nj = np.sum(idx) 
      coeff = 1.0/(nj-1.0)
      
      #use pairwise cluster calc
      pts = X[idx]
      dists = np.log(sp.spatial.distance.pdist(pts))
      distSum = np.sum(dists)
     
      #trying to use cached distances - this slicing is slower
      #idx = self.labels_==j
      #distSum = np.sum(self.dists_[:,idx][idx])/2

      #score sum
      score += coeff*distSum
    return score  
  
#Gaussian ring and center blob example
if __name__ == "__main__":
  N=100
  
  #gen ring samples
  ring = eval.noisy_ring(N, (0,0), 10.0)
  ringTrue = [1]*N
  
  #gen norm samples
  means = np.array([0, 0]); 
  cov   = np.matrix([[1,0], [0,1]])
  norm  = np.random.multivariate_normal(means, cov, N)
  normTrue = [0]*N

  #combine samps
  X = np.concatenate( (norm, ring) ) 
  labels_true = np.concatenate( (normTrue,ringTrue) )

  #run NIC clustering algo - need to report score somehow
  nic = NIC()
  nic.fit(X)

if True:
  #plot results
  cluster0 = X[nic.labels_==0]
  cluster1 = X[nic.labels_==1]
  plt.scatter(cluster0[:,0], cluster0[:,1], marker='^', c='r')
  plt.scatter(cluster1[:,0], cluster1[:,1], marker="^", c="b")
  plt.show()
