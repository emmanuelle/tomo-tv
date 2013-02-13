""" LV + l1 proximal operator

The core idea here is to modify the analysis operator in the Beck &
Teboulle approach (actually Chambolle) to keep the identity and thus to
end up with an l1.
"""

import numpy as np


def div_id(grad, l1_ratio=1.):
    """ Compute divergence + id of image gradient + id"""
    res = np.zeros(grad.shape[1:])
    # The divergence part
    for d in range(grad.shape[0] - 1):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    # The identity part
    res -= l1_ratio * grad[-1]
    return res


def gradient_id(img, l1_ratio=1.):
    """
    Compute gradient + id of an image

    Parameters
    ===========
    img: ndarray
        N-dimensional image

    Returns
    =======
    gradient: ndarray
        Gradient of the image: the i-th component along the first
        axis is the gradient along the i-th axis of the original
        array img
    """
    shape = [img.ndim + 1, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    # The gradient part: 'Clever' code to have a view of the gradient
    # with dimension i stop at -1
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    # The identity part
    gradient[-1] = l1_ratio * img
    return gradient


def _projector_on_dual(grad):
    """
    modifies IN PLACE the gradient + id to project it
    on the l21 unit ball in the gradient direction and the l1 ball in the
    identity direction
    """
    # The l21 ball for the gradient direction
    norm = np.sqrt(np.sum(grad[:-1]**2, 0))
    norm.clip(1., out=norm)
    for grad_comp in grad[:-1]:
        grad_comp /= norm
    # The l1 ball for the identity direction
    norm = np.abs(grad[-1])
    norm.clip(1., out=norm)
    grad[-1] /= norm
    return grad


def total_variation(gradient):
    """ Our total-variation like norm
    """
    return (np.sum(np.sqrt(np.sum(gradient[:-1]**2, axis=0)))
            + np.sum(np.abs(gradient[-1])))


def dual_gap(input_img_norm, new, gap, weight, l1_ratio=1.):
    """
    dual gap of total variation denoising
    see "Total variation regularization for fMRI-based prediction of behavior",
    by Michel et al. (2011) for a derivation of the dual gap
    """
    tv_new = total_variation(gradient_id(new, l1_ratio=l1_ratio))
    d_gap = (gap**2).sum() + 2*weight*tv_new - input_img_norm + (new**2).sum()
    return 0.5 / input_img_norm * d_gap


def _objective_function(input_img, output_img, gradient, weight):
    return (.5 * ((input_img - output_img)**2).sum()
            + weight * total_variation(gradient))


@profile
def tv_l1_fista(im, l1_ratio=.05, weight=50, dgap_tol=5.e-5, x_tol=None,
                     n_iter_max=200,
                     check_gap_frequency=4, val_min=None, val_max=None,
                     verbose=True, fista=True):
    """
    Perform total-variation denoising on 2-d and 3-d images

    Find the argmin `res` of
        1/2 * ||im - res||^2 + weight * TV(res),

    where TV is the isotropic l1 norm of the gradient.

    Parameters
    ----------
    im: ndarray of floats (2-d or 3-d)
        input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    weight: float, optional
        denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    dgap_tol: float, optional
        precision required. The distance to the exact solution is computed
        by the dual gap of the optimization problem and rescaled by the
        squared l2 norm of the image (for contrast invariance).

    x_tol: float or None, optional
        The maximal relative difference between input and output. If
        specified, this specifies a stopping criterion on x, rather than
        the dual gap

    n_iter_max: int, optional
        maximal number of iterations used for the optimization.

    val_min: None or float, optional
        an optional lower bound constraint on the reconstructed image

    val_max: None or float, optional
        an optional upper bound constraint on the reconstructed image

    verbose: bool, optional
        if True, plot the dual gap of the optimization

    Returns
    -------
    out: ndarray
        denoised array

    Notes
    -----
    The principle of total variation denoising is explained in
    http://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This function implements the FISTA (Fast Iterative Shrinkage
    Thresholding Algorithm) algorithm of Beck et Teboulle, adapted to
    total variation denoising in "Fast gradient-based algorithms for
    constrained total variation image denoising and deblurring problems"
    (2009).

    For details on implementing the bound constraints, read the Beck and
    Teboulle paper.
    """
    weight = float(weight)
    input_img = im
    input_img_norm = (im ** 2).sum()
    if not input_img.dtype.kind == 'f':
        input_img = input_img.astype(np.float)
    shape = [input_img.ndim + 1, ] + list(input_img.shape)
    grad_im = np.zeros(shape)
    grad_aux = np.zeros(shape)
    t = 1.
    i = 0
    if input_img.ndim == 2:
        # Upper bound on the Lipschitz constant
        # Theory tells us that the Lipschitz constant of div * grad is 8
        lipschitz_constant = 1.1 * (8 + l1_ratio**2)
    elif input_img.ndim == 3:
        # Theory tells us that the Lipschitz constant of div * grad is 12
        lipschitz_constant = 1.1 * (12 + l1_ratio**2)
    else:
        raise ValueError('Cannot compute TV for images that are not '
                         '2D or 3D')
    # negated_output is the negated primal variable in the optimization
    # loop
    negated_output = -input_img
    # Clipping values for the inner loop
    negated_val_min = np.inf
    negated_val_max = -np.inf
    if val_min is not None:
        negated_val_min = -val_min
    if val_max is not None:
        negated_val_max = -val_max
    if True or (val_min is not None or val_max is not None):
        # With bound constraints, the stopping criterion is on the
        # evolution of the output
        negated_output_old = negated_output.copy()
    grad_tmp = None
    while i < n_iter_max:
        grad_tmp = gradient_id(negated_output, l1_ratio=l1_ratio)
        grad_tmp *= 1. / (lipschitz_constant * weight)
        grad_aux += grad_tmp
        grad_tmp = _projector_on_dual(grad_aux)
        # Carefull, in the next few lines, grad_tmp and grad_aux are a
        # view on the same array, as _projector_on_dual returns a view
        # on the input array
        t_new = 1. / 2 * (1 + np.sqrt(1 + 4 * t**2))
        t_factor = (t - 1) / t_new
        if fista:
            grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im
        else:
            grad_aux = grad_tmp
        grad_im = grad_tmp
        t = t_new
        gap = weight * div_id(grad_aux, l1_ratio=l1_ratio)
        # Compute the primal variable
        negated_output = gap - input_img
        if (val_min is not None or val_max is not None):
            negated_output = negated_output.clip(negated_val_max,
                                negated_val_min,
                                out=negated_output)
        if (i % check_gap_frequency) == 0:
            if x_tol is None:
                # Stopping criterion based on the dual_gap
                if val_min is not None or val_max is not None:
                    # We need to recompute the dual variable
                    gap = negated_output + input_img
                dgap = dual_gap(input_img_norm, -negated_output,
                                gap, weight, l1_ratio=l1_ratio)
                if verbose:
                    print 'Iteration % 2i, dual gap: % 6.3e' % (i, dgap)
                if dgap < dgap_tol:
                    break
            else:
                # Stopping criterion based on x_tol
                diff = np.max(np.abs(negated_output_old - negated_output))
                diff /= np.max(np.abs(negated_output))
                if verbose:
                    print ('Iteration % 2i, relative difference: % 6.3e,'
                           'energy: % 6.3e' % (i, diff,
                           _objective_function(input_img,
                                -negated_output,
                                gradient_id(negated_output, l1_ratio=l1_ratio),
                                weight)))
                if diff < x_tol:
                    break
                negated_output_old = negated_output
        i += 1
    # Compute the primal variable, however, here we must use the ista
    # value, not the fista one
    output = input_img - weight * div_id(grad_im, l1_ratio=l1_ratio)
    if (val_min is not None or val_max is not None):
        output = output.clip(val_min, val_max, out=output)
    return output


def test_grad_div_adjoint(size=12, random_state=42):
    # We need to check that <D x, y> = <x, DT y> for x and y random vectors
    random_state = np.random.RandomState(random_state)

    x = np.random.normal(size=(size, size, size))
    y = np.random.normal(size=(4, size, size, size))

    np.testing.assert_almost_equal(np.sum(gradient_id(x, l1_ratio=2.) * y),
                                   -np.sum(x * div_id(y, l1_ratio=2.)))


if __name__ == '__main__':
    # First our test
    test_grad_div_adjoint()
    import matplotlib.pyplot as plt
    from time import time

    np.random.seed(0)
    l = np.zeros((256, 256))
    l[10:50, 10:50] = 1.2
    l[-60:-30, 120:220] = 2.
    l_noisy = l + 3 * l.std() * np.random.randn(*l.shape)
    t0 = time()
    res = tv_l1_fista(l, weight=2.5, l1_ratio=.02, dgap_tol=1.e-5,
                      verbose=True, fista=True, n_iter_max=5000)
    t1 = time()
    print t1 - t0
    plt.figure()
    plt.subplot(121)
    plt.imshow(l, cmap=plt.cm.gist_earth, vmin=-1., vmax=5.)
    plt.subplot(122)
    plt.imshow(res, cmap=plt.cm.gist_earth, vmin=-1., vmax=5.)
    plt.show()


