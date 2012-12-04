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

from sklearn.linear_model import Ridge

# Synthetic data
l = 256
x = generate_synthetic_data(l)

n_iter = 150

# Projection operator and projections data, with noise
H = build_projection_operator(l, 40)
y = H * x.ravel()[:, np.newaxis]
y += 2*np.random.randn(*y.shape)

# Reconstruction
t1 = time()
# Penalty parameter hand-tuned to minimize error
res, energies = fista_haar(y, 0.28, n_iter, H, level=None)
t2 = time()
print "reconstruction done in %f s" %(t2 - t1)

l2_estimate = Ridge().fit(H, y).coef_.ravel()

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

plt.figure(figsize=(8, 3.3), edgecolor='k', facecolor='k')
plt.subplot(131)
plt.imshow(x, cmap=plt.cm.gnuplot2, interpolation='nearest', vmin=-.1)
plt.axis('off')
plt.title('Original image', color='w')
plt.subplot(132)
plt.imshow(np.reshape(l2_estimate, x.shape),
           cmap=plt.cm.gnuplot2, interpolation='nearest',
           vmin=-.2)
plt.title('Non-sparse reconstruction', color='w')
plt.axis('off')
plt.subplot(133)
plt.imshow(res[-1], cmap=plt.cm.gnuplot2, interpolation='nearest',
           vmin=-.2)
plt.title('Sparse reconstruction', color='w')
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)

plt.savefig('haar_tomo.svg', edgecolor='k', facecolor='k')
plt.savefig('haar_tomo.pdf', edgecolor='k', facecolor='k')

plt.figure(figsize=(3.3, 3.3), edgecolor='k', facecolor='k')
error = x - res[-1]
vmax = np.abs(error).max()
plt.imshow(error, cmap=plt.cm.gnuplot2, interpolation='nearest',
            vmin=-vmax, vmax=vmax)
plt.axis('off')

plt.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                    right=1)

print 'Error norm: %.3e' % np.sqrt(np.sum(error **2))


plt.show()
