import numpy as np
import ContinuousMarkov_lib as mdl

#Durrett P165 Example 4.14 (Duke Basketball)
mu = np.array([4.0, 15/2, 3.0, 10.0]) #サービス率
p = np.array([[0.0, 0.7, 0.3, 0.0],[0.0, 0.0, 1.0, 0.0],[1/3, 0.0, 0.0, 2/3], [1.0, 0.0, 0.0, 0.0]])

clib = mdl.ContinuousMarkov_lib(p, mu)
q = clib.getTransitionRate()
print(q)
pi = clib.getStationary()
print(pi)