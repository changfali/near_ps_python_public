import imp
from scipy import sparse as sp
import numpy as np
A = np.load('test/A.npy')
A = np.vstack([A for i_ in range(3)])   
print(A[0,0],A[3200,0])