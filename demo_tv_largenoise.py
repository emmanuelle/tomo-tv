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
from reconstruction.forward_backward_tv import fista_tv, gfb_tv
from reconstruction.projections import build_projection_operator
from reconstruction.util import generate_synthetic_data
import matplotlib.pyplot as plt

# Synthetic data
l = 256
x = generate_synthetic_data(l)



# Projection operator and projections data, with noise
H = build_projection_operator(l, 40)
y = H * x.ravel()[:, np.newaxis]
y += 10*np.random.randn(*y.shape)

# Reconstruction
res_tv, energies_tv = fista_tv(y, 120, 300, H)
res_tv_compact, energies_tv_compact = gfb_tv(y, 120, 500, H, stop_tol=1.e-10)

# Fraction of errors of segmented image wrt ground truth
err_tv = [np.abs(x - (resi > 0.5)).mean() for resi in res_tv]
err_tv_compact = [np.abs(x - (resi > 0.5)).mean() for resi in res_tv_compact]

error_tv = x - res_tv[-1]
error_tv_compact = x - res_tv_compact[-1]
print 'Error norm: %.3e' % np.sqrt(np.sum(error_tv **2))
print 'Error norm: %.3e' % np.sqrt(np.sum(error_tv_compact **2))

err_max = np.abs(error_tv).max()
err_max_compact = np.abs(error_tv_compact).max()

# Display results
plt.figure(figsize=(11, 4))
plt.subplot(131)
plt.imshow(x, cmap='gnuplot2', interpolation='nearest', vmin=-0.1, vmax=1)
plt.title('original data', fontsize=20)
plt.axis('off')
plt.subplot(132)
plt.imshow(error_tv, cmap='gnuplot2', interpolation='nearest',
                vmin=-err_max, vmax=err_max)
plt.title('error for TV penalization', fontsize=20)
plt.axis('off')
plt.subplot(133)
plt.imshow(error_tv_compact, cmap='gnuplot2', interpolation='nearest',
                vmin=-err_max, vmax=err_max)
plt.title('error for TV + compact', fontsize=20)
plt.axis('off')
plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)

plt.show()

plt.savefig('tv_vs_tv+compact.pdf')
plt.savefig('tv_vs_tv+compact.svg')
plt.savefig('tv_vs_tv+compact.png')
