"""
Total-variation penalization for tomography reconstruction
==========================================================

In this example, we reconstruct an image from its tomography projections
with an uncomplete set of projections (l/3 angles, where l is the linear
size of the image. For a correct reconstruction without a-priori information,
one would usually require l or more angles). In addition, noise is added to
the projections.

In order to reconstruct the original image, we minimize a function that is
the sum of (i) a L2 data fit term, and (ii) the total variation of the image.
Proximal iterations using the FISTA scheme are used.

This example should take around 30 seconds to compute and plot the results.
"""

print __doc__

import numpy as np
from reconstruction.forward_backward_tv import fista_tv
from reconstruction.projections import build_projection_operator
from reconstruction.util import generate_synthetic_data
from time import time
import matplotlib.pyplot as plt

# Synthetic data
l = 256
x = generate_synthetic_data(l)



# Projection operator and projections data, with noise
H = build_projection_operator(l, 40)
y = H * x.ravel()[:, np.newaxis]
y += 2*np.random.randn(*y.shape)

# Reconstruction
res, energies = fista_tv(y, 10, 300, H) 



# Fraction of errors of segmented image wrt ground truth
err = [np.abs(x - (resi > 0.5)).mean() for resi in res]

error = x - res[-1]
print 'TV error norm: %.3e' % np.sqrt(np.sum(error **2))

haar = np.load('haar.npy')
err_haar = x - haar
print 'Haar error norm: %.3e' % np.sqrt(np.sum(err_haar **2))

# Clip the gray value interval to enhance contrast
err_max = 0.7 * np.abs(err_haar).max()

# Display results
plt.figure(figsize=(11, 4))
plt.subplot(131)
plt.imshow(x, cmap='gnuplot2', interpolation='nearest', vmin=-0.1, vmax=1)
plt.title('original data', fontsize=20)
plt.axis('off')
plt.subplot(132)
plt.imshow(err_haar, cmap='gnuplot2', interpolation='nearest',
                vmin=-err_max, vmax=err_max)
plt.title('error for Haar wavelet', fontsize=20)
plt.axis('off')
plt.subplot(133)
plt.imshow(error, cmap='gnuplot2', interpolation='nearest',
                vmin=-err_max, vmax=err_max)
plt.title('error for TV penalization', fontsize=20)
plt.axis('off')
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

plt.show()

plt.savefig('tv_vs_haar.png')
plt.savefig('tv_vs_haar.pdf')
plt.savefig('tv_vs_haar.svg')
