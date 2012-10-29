"""
In this example, we try to determine the best value of the parameter beta
with cross-validation while reconstructing images from their tomographic
projections with a total-variation regularization of weight beta.

Cross-validation is a very useful method when one does not know the ground
truth.

For a given realization of the noise, we can reconstruct the image from the
noisy measurements for several values of beta. For each of these solutions, we
can compute the l2 distance between its set of projections, and an another set
of measures corresponding to a different realization of the noise. Since we
expect the errors to be uncorrelated, the minimum distance corresponds to the
reconstruction closest to the initial image.

In the example below, we observe for a binary image that using this
cross-validation method, we select the value of beta that minimizes as well
the segmentation error.
"""
print __doc__

import numpy as np
from reconstruction.util import generate_synthetic_data
from reconstruction.projections import build_projection_operator
from reconstruction.forward_backward_tv import fista_tv
from joblib import Parallel, delayed

# Synthetic data
l = 256
x = generate_synthetic_data(l, n_pts=100)

n_dir = 80


# Projection operator and projections data, with 2 realizations of the noise
H = build_projection_operator(l, n_dir)
y = H * x.ravel()[:, np.newaxis]
np.random.seed(0)
y1 = y + 2*np.random.randn(*y.shape)  # 1st realization
y2 = y + 2*np.random.randn(*y.shape)  # 2nd realization

# Range of beta parameter
betas = 2**np.arange(2, 6, 0.25)


def rec_error(beta):
    """
    cross-validation
    """
    res, energies = fista_tv(y1, beta, 400, H)
    yres = H * res[-1].ravel()[:, np.newaxis]
    return (((yres - y2)**2).mean()), res[-1], energies


results = Parallel(n_jobs=-1)(delayed(rec_error)(beta) for beta in betas)

errors = [res[0] for res in results]

images = [res[1] for res in results]

energies = [res[2] for res in results]

# Segmentation compared to ground truth
segmentation_error = [np.abs((image > 0.5) - x).mean() for image in images]


print "best beta from cross-validation %f" %(betas[np.argmin(errors)])
print "best beta for segmentation compared to ground truth %f" \
                    %(betas[np.argmin(segmentation_error)])
