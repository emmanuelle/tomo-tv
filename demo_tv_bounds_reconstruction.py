"""
Total-variation penalization and bound constraints for tomography reconstruction
================================================================================

In this example, we reconstruct an image from its tomography projections
with an uncomplete set of projections (l/8 angles, where l is the linear
size of the image. For a correct reconstruction without a-priori information,
one would usually require l or more angles). In addition, noise is added to
the projections.

In order to reconstruct the original image, we minimize a function that
is the sum of (i) a L2 data fit term, and (ii) the total variation of the
image, and bound constraints on the pixel values. Proximal iterations
using the FISTA scheme are used.

We compare with and without the bounds

This example should take around 1mn to run and plot the results.
"""

print __doc__

import numpy as np
from reconstruction.forward_backward_tv import fista_tv
from reconstruction.projections import build_projection_operator
from reconstruction.util import generate_synthetic_data
from time import time
import matplotlib.pyplot as plt

# Synthetic data
l = 512
np.random.seed(0)
x = generate_synthetic_data(l)


# Projection operator and projections data, with noise
H = build_projection_operator(l, l / 32)
y = H * x.ravel()[:, np.newaxis]
y += 5 * np.random.randn(*y.shape)

# Display original data
plt.figure(figsize=(12, 5))
plt.subplot(2, 3, 1)
plt.imshow(x, cmap=plt.cm.gnuplot2, interpolation='nearest', vmin=-.1, vmax=1.2)
plt.title('original data (256x256)')
plt.axis('off')

for idx, (val_min, val_max, name) in enumerate([
                                        (None, None, 'TV'),
                                        (0, 1, 'TV + interval'),
                                    ]):
    # Reconstruction
    t1 = time()
    res, energies = fista_tv(y, 50, 100, H, val_min=val_min,
                             val_max=val_max)
    t2 = time()

    # Fraction of errors of segmented image wrt ground truth
    err = np.abs(x - (res[-1] > 0.5)).mean()
    print "%s: reconstruction done in %f s, %.3f%% segmentation error" % (
                name, t2 - t1, 100 * err)

    plt.subplot(2, 3, 2 + idx)
    plt.imshow(res[-1], cmap=plt.cm.gnuplot2, interpolation='nearest', vmin=-.1,
                vmax=1.2)
    plt.title('reconstruction with %s' % name)
    plt.axis('off')
    ax = plt.subplot(2, 3, 5 + idx)
    ax.yaxis.set_scale('log')
    plt.hist(res[-1].ravel(), bins=20, normed=True)
    plt.yticks(())
    plt.title('Histogram of pixel intensity')
    plt.axis('tight')

plt.show()
