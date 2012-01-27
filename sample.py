import numpy as np

def noisy_ring(N, center, radius, var=1.0):
  """Creates N 2d samples from a (gaussian) noisy ring"""
  thetas = np.random.uniform(0, 2*np.pi, N)
  xnoise = np.random.normal(0, var, N)
  ynoise = np.random.normal(0, var, N)
  points = np.empty( (N, 2) )
  points[:,0] = radius*np.cos(thetas) + xnoise + center[0]
  points[:,1] = radius*np.sin(thetas) + ynoise + center[1]
  return points  

