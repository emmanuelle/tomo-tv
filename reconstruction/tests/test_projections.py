import numpy as np
from ..projections import build_projection_operator, projection, \
                back_projection, filter_projections
from numpy.testing import assert_almost_equal

def test_build_projection_operator():
    l_x = 16
    n_dir = l_x
    op =  build_projection_operator(l_x, n_dir=n_dir)
    assert op.data.max() == 1.
    assert op.shape == (l_x**2, l_x**2)
    # Number of pixels along the diagonal: max nb of pixels
    # contributing to a pixel
    assert_almost_equal(op.sum(axis=1).max(), l_x*np.sqrt(2), decimal=-1)
    # Contributions of a single pixel
    assert_almost_equal(op.sum(axis=0).max(), n_dir, decimal=4)
    op = build_projection_operator(l_x, n_dir=n_dir, subpix=2)
    assert_almost_equal(op.sum(axis=1).max(), l_x*np.sqrt(2), decimal=-1)
    assert_almost_equal(op.sum(axis=0).max(), n_dir, decimal=4)

def test_multiscale_projection_operator():
    l_x = 16
    l_det = 2 * l_x
    n_dir = l_x
    op = build_projection_operator(l_x, n_dir=n_dir, l_det=l_det, subpix=2)
    assert op.shape == (l_det * n_dir, l_x**2)
    assert_almost_equal(op.sum(axis=0).max(), n_dir, decimal=4)
    # One big data pixel is projected onto several detector pixels
    assert op.data.max() > 0.5 and op.data.max() < 1

def test_object_larger_than_detector_operator():
    l_x = 32
    l_det = 32
    offset = 8
    n_dir = 32
    op = build_projection_operator(l_x, l_det=l_det, n_dir=l_x, subpix=2, 
                                offset=offset)
    assert op.data.max() > 0.5 and op.data.max() < 1
    X, Y = np.ogrid[:l_x, :l_x]
    mask_full = ((X - l_x/2)**2 + (Y - l_x/2)**2 < \
                                    0.9 * ((l_x - 2 * offset)/2)**2).ravel()
    # Image pixels that contribute to all projections (in the center)
    assert np.all(op.sum(axis=0)[:, mask_full] > 0.9 * n_dir)

def test_build_operator_with_mask():
    l_x = 32
    n_dir = 16
    X, Y = np.ogrid[:l_x, :l_x]
    mask = (X - l_x/2)**2 + (Y - l_x/2)**2 < 0.9 * (l_x / 2)**2
    op = build_projection_operator(l_x, n_dir=n_dir, pixels_mask=mask)
    assert op.shape[1] == mask.sum()
    assert np.all(op.sum(axis=0) > 0.9 * n_dir)


def test_direct_projection():
    l_x = 16
    n_dir = l_x / 2
    im = np.zeros((l_x, l_x))
    projections = projection(im, n_dir=n_dir)
    assert projections.shape == (n_dir, l_x)
    assert np.all(projections.sum(axis=1) == im.sum())

def test_compare_projection_methods():
    l_x = 16
    n_dir = l_x / 2
    im = np.zeros((l_x, l_x))
    im[l_x / 3:-l_x / 4, l_x / 3: -l_x / 4] = 1
    proj_direct_lin = projection(im, n_dir=n_dir, interpolation='linear')
    proj_direct_nn = projection(im, n_dir=n_dir, interpolation='nearest')
    op = build_projection_operator(l_x, n_dir=n_dir)
    proj_matrix = (op * im.ravel()[:, np.newaxis]).reshape((n_dir, l_x))
    assert np.all(proj_matrix == proj_direct_lin)
    # The error between linear and nearest neighbor interpolation is 
    # quite large for a very small image
    assert np.abs(proj_direct_nn - proj_direct_lin).mean() \
        / np.abs(proj_direct_lin).mean() < 0.2

def test_projection_back_projection():
    l_x = 16
    n_dir = l_x / 2
    im = np.zeros((l_x, l_x))
    im[l_x / 3:-l_x / 4, l_x / 3: -l_x / 4] = 1
    projections = projection(im, n_dir=n_dir, interpolation='linear')
    reconstruction = back_projection(projections)
    assert reconstruction.shape == im.shape

def test_projection_filtered_back_projection():
    l_x = 16
    n_dir = 2 * l_x
    im = np.zeros((l_x, l_x))
    im[l_x / 3:-l_x / 4, l_x / 3: -l_x / 4] = 1
    projections = projection(im, n_dir=n_dir, interpolation='linear')
    filtered_projections = filter_projections(projections)
    reconstruction = back_projection(projections)
    assert reconstruction.shape == im.shape
