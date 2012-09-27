import numpy as np

from ..projections import build_projection_operator
from ..util import generate_synthetic_data
from ..forward_backward_tv import ista_tv, fista_tv, gfb_tv
from scipy import ndimage

def test_proximal_iterations():
    l_x = 64
    x = generate_synthetic_data(l_x)
    H = build_projection_operator(l_x, n_dir=l_x / 3)
    y = H * x.ravel()[:, np.newaxis]
    res1, energies1 = fista_tv(y, 50, 5, H)
    res2, energies2 = ista_tv(y, 50, 5, H)
    res3, energies3 = ista_tv(y.reshape((l_x/3, l_x)), 50, 5)
    assert np.all(np.diff(energies1) <= 0)
    assert np.abs(np.diff(energies1[:5])).mean() > \
                np.abs(np.diff(energies2)).mean()
    assert np.allclose(energies2, energies3, rtol=8.e-2)

def test_proximal_gfb():
    l_x = 64
    x = generate_synthetic_data(l_x)
    H = build_projection_operator(l_x, n_dir=l_x / 3)
    y = H * x.ravel()[:, np.newaxis]
    np.random.seed(0)
    y += 2*np.random.randn(*y.shape)
    res1, energies1 = gfb_tv(y, 50, 10, H)
    res2, energies2 = ista_tv(y, 50, 10, H)
    assert np.all(np.diff(energies1) <= 0)
    assert np.sum((res1[-1] - x)**2) < np.sum((res2[-1] - x)**2)
    assert np.max(res1[-1]) < 1.02
    assert np.min(res1[-1]) > -0.02


def test_proximal_iterations_with_mask():
    l_x = 64
    x = generate_synthetic_data(l_x)
    X, Y = np.ogrid[:l_x, :l_x]
    mask = (X - l_x/2)**2 + (Y - l_x/2)**2 < (l_x / 2)**2
    H = build_projection_operator(l_x, n_dir=l_x / 3, pixels_mask=mask)
    y = H * x[mask][:, np.newaxis]
    res1, energies1 = fista_tv(y, 50, 10, H, mask=mask)
    assert np.all(np.diff(energies1) <= 0)
    assert np.all(np.abs(res1[-1][~ndimage.binary_dilation(mask, iterations=3)]) < 0.011)
