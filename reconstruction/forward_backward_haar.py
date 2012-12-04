import numpy as np
import pywt
from projections import back_projection, projection
from scipy import sparse


# ------------------ Proximal iterators ----------------------------

def fista_haar(y, beta, niter, A, level=None):
    """
    Haar-wavelet l1 regularization using FISTA algorithm
    (Fast Iterative Shrinkage/Thresholding Algorithm)

    Parameters
    ----------

    y : ndarray of floats
        Measures (tomography projection). If H is given, y is a column
        vector. If H is not given, y is a 2-D array where each line
        is a projection along a different angle

    beta : float
        weight of Haar l1 norm

    niter : number of forward-backward iterations to perform

    A : sparse matrix
        tomography design matrix. Should be in csr format.


    Returns
    -------

    res : list
        list of iterates of the reconstructed images

    energies : list
        values of the function to be minimized at the different
        iterations. Its values should be decreasing.

    Notes
    -----
    This algorithm minimizes iteratively the energy

    E(w) = 1/2 || A Ht w - y ||^2 + beta * l1(w) = f(x) + beta * l1(w)

    where w are Haar wavelet coefficients, Ht is the inverse wavelet.
    transformation operator, A is the tomography projection operator

    References
    ----------

    - A. Beck and M. Teboulle (2009). A fast iterative
      shrinkage-thresholding algorithm for linear inverse problems.
      SIAM J. Imaging Sci., 2(1):183-202.
    """
    n_meas, n_pix = A.shape
    l = int(np.sqrt(n_pix))
    n_angles = n_meas / l
    At = sparse.csr_matrix(A.transpose())
    x0 = np.zeros((l, l))
    if level is None:
        level = int(np.log2(l))
    coefs = pywt.wavedec2(x0, 'haar', level=level)
    res, energies = [], []
    gamma = .9/ (l * n_angles)
    x = x0
    u_old = coefs[:]
    t_old = 1
    results = []
    for i in range(niter):
        x = pywt.waverec2(coefs, 'haar')
        results.append(x)
        x = x.ravel()[:, np.newaxis]
        err = A * x - y
        back_proj = At * err
        coefs_backproj = pywt.wavedec2(back_proj.reshape((l, l)),
                                    'haar', level=level)
        coefs_tmp = []
        for coef_arrays, coef_arrays_bp in zip(coefs, coefs_backproj):
            res = []
            for coef_array, coef_array_bp in \
                        zip(coef_arrays, coef_arrays_bp):
                res.append(coef_array - gamma * coef_array_bp)
            coefs_tmp.append(res)
        u_n = soft_thresholding_haar(coefs_tmp, weight=beta*gamma)
        t_new = (1 + np.sqrt(1 + 4 * t_old**2))/2.
        t_old = t_new
        coefs = []
        for u_n_arrays, u_old_arrays in zip(u_n, u_old):
            res = []
            for u_n_array, u_old_array in \
                        zip(u_n_arrays, u_old_arrays):
                res.append(u_n_array +
                    (t_old - 1)/t_new * (u_n_array - u_old_array))
            coefs.append(res)
        u_old = u_n
        data_fidelity_err = 1./2 * (err**2).sum()
        l1_norm = beta * l1_norm_haar(coefs)
        energy = data_fidelity_err + l1_norm
        print energy
        energies.append(energy)
    return results, energies


def ista_haar(y, beta, niter, A=None):
    """
    Haar-wavelet l1 regression using ISTA algorithm
    (Iterative Shrinkage/Thresholding Algorithm)

    Parameters
    ----------

    y : ndarray of floats
        Measures (tomography projection). If H is given, y is a column
        vector. If H is not given, y is a 2-D array where each line
        is a projection along a different angle

    beta : float
        weight of Haar l1 norm

    niter : number of forward-backward iterations to perform

    A : sparse matrix or None
        tomography design matrix. Should be in csr format. If H is none,
        the projections as well as the back-projection are computed by
        a direct method, without writing explicitely the design matrix.

    Returns
    -------

    res : list
        list of iterates of the reconstructed images

    energies : list
        values of the function to be minimized at the different
        iterations. Its values should be decreasing.

    Notes
    -----

    This algorithm minimizes iteratively the energy

    E(x) = 1/2 || H x - y ||^2 + beta TV(x)

    by simple forward - backward iterations:

    x_{n + 1} = prox_{gamma beta TV(.)} (x_n - gamma nabla f(x_n))

    where f(x) = 1/2 || H x - y ||^2

    References
    ----------

    - Proximal Splitting Methods in Signal Processing, P. Combettes
      and J.-C. Pesquet, Fixed-Point Algorithms for Inverse Problems
      in Science and Engineering, p. 185 (2011). Algorithm 10.3 with
      lamba_n = 1.

    - Nelly Pustelnik's thesis (in French),
      http://tel.archives-ouvertes.fr/tel-00559126_v4/
      Paragraph 3.3.1-c p. 68 , ISTA

    """
    if A is None:
        method = 'direct'
        n_angles, l = y.shape
        n_pix = l ** 2
    else:
        method = 'matrix'
        n_angles, l = y.shape
        n_meas, n_pix = A.shape
        l = int(np.sqrt(n_pix))
        n_angles = n_meas / l
    if method == 'matrix':
        At = sparse.csr_matrix(A.transpose())
    x0 = np.zeros((l, l))
    coefs = pywt.wavedec2(x0, 'haar')
    res, energies = [], []
    # l * n_angles is the Lipschitz constant of Ht H
    gamma = .9/ (l * n_angles)
    x = x0
    results = []
    for i in range(niter):
        # Forward part
        x = pywt.waverec2(coefs, 'haar')
        results.append(x)
        if method == 'matrix':
            x = x.ravel()[:, np.newaxis]
            err = A * x - y
            back_proj = At * err
            coefs_backproj = pywt.wavedec2(back_proj.reshape((l, l)), 'haar')
            coefs_tmp = []
            for coef_arrays, coef_arrays_bp in zip(coefs, coefs_backproj):
                res = []
                for coef_array, coef_array_bp in \
                            zip(coef_arrays, coef_arrays_bp):
                    res.append(coef_array - gamma * coef_array_bp)
                coefs_tmp.append(res)
        else:
            err = projection(x, n_angles) - y
            back_proj = back_projection(err)
            coefs_backproj = pywt.wavedec2(back_proj.reshape((l, l)), 'haar')
            coefs_tmp = []
            for coef_arrays, coef_arrays_bp in zip(coefs, coefs_backproj):
                res = []
                for coef_array, coef_array_bp in \
                            zip(coef_arrays, coef_arrays_bp):
                    res.append(coef_array - gamma * coef_array_bp)
                coefs_tmp.append(res)
        # backward: TV prox
        coefs = soft_thresholding_haar(coefs_tmp, weight=beta*gamma)
        # compute the energy
        data_fidelity_err = 1./2 * (err**2).sum()
        l1_norm = beta * l1_norm_haar(coefs)
        energy = data_fidelity_err + l1_norm
        print energy
        energies.append(energy)
    return results, energies


def l1_norm_haar(coefs):
    l1_norm = 0
    for coef_arrays in coefs:
        for coef_array in coef_arrays:
            l1_norm += np.abs(coef_array).sum()
    return l1_norm


def soft_thresholding_haar(coefs, weight=1):
    # This soft thresholding operator scales the weight down in low
    # levels (low freq info), as there is less noise and the coefs are
    # less sparse
    res = []
    for level, coef_arrays in enumerate(coefs):
        new_coefs = []
        this_weight = weight * (level + 1) ** 2
        for coef_array in coef_arrays:
            new_coefs.append(np.sign(coef_array) * \
                             np.maximum(0, np.abs(coef_array) - this_weight))
        res.append(new_coefs)
    return res
