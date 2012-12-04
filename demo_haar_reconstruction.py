"""
Haar-wavelet l1 regularization for tomography reconstruction
============================================================
"""

print __doc__

import numpy as np
from reconstruction.forward_backward_haar import fista_haar, ista_haar
from reconstruction.projections import build_projection_operator
from reconstruction.util import generate_synthetic_data
from time import time
import matplotlib.pyplot as plt

# Synthetic data
l = 256
x = generate_synthetic_data(l)

n_iter = 150

# Projection operator and projections data, with noise
H = build_projection_operator(l, 80)
y = H * x.ravel()[:, np.newaxis]
y += 2*np.random.randn(*y.shape)

# Reconstruction
t1 = time()
res, energies = fista_haar(y, 0.1, n_iter, H, level=None) 
t2 = time()
print "reconstruction done in %f s" %(t2 - t1)

# Fraction of errors of segmented image wrt ground truth
err = [np.abs(x - (resi > 0.5)).mean() for resi in res]

# Display results
plt.figure()
plt.subplot(221)
plt.imshow(x, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
plt.title('original data (256x256)')
plt.axis('off')
plt.subplot(222)
plt.imshow(res[-1], cmap='gray', interpolation='nearest', vmin=0, vmax=1)
plt.title('reconstruction after %d iterations' %n_iter)
plt.axis('off')
plt.subplot(223)
plt.loglog(energies, 'o')
plt.xlabel('iteration number')
plt.title('energy')
plt.subplot(224)
plt.loglog(err, 'o')
plt.xlabel('iteration number')
plt.title('error fraction')
plt.show()
