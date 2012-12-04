import numpy as np
from scipy import signal
from scipy import sparse
from reconstruction.projections import build_projection_operator
from reconstruction.util import generate_synthetic_data
import matplotlib.pyplot as plt

def find_close_function(im, n_angles, beta=100, n_iter=100):
    l = len(im)
    gamma = .9/ (l * n_angles + 2*beta)
    x = np.copy(im)
    X, Y = np.ogrid[:l, :l]
    mask = (X - l/2.)**2 + (Y - l/2.)**2 < (l/2.)**2
    H = build_projection_operator(l, n_angles, pixels_mask=mask)
    Ht = sparse.csr_matrix(H.transpose())
    energies = []
    for i in range(n_iter):
        dx = gamma * (Ht * H * x[mask][:, None]).ravel()
        x[mask] -= dx
        x -= gamma * beta * (x - im)
        res = H * x[mask][:, None]
        energies.append(((res**2).sum()) + beta/2*((x - im)**2).sum())
        print i, energies[-1]
    res = Ht * H * x[mask][:, None]
    return x, (res**2).sum() / (x[mask]**2).sum()

l = 128

im_nb = 2

if im_nb == 1:
    thet = np.pi/16.
    x, y = np.ogrid[:l, :l]
    mask = (x - l/2.)**2 + (y - l/2.)**2 < (l/2.)**2
    s = np.sin(np.pi/4 * (np.cos(thet)*x  - np.sin(thet)*y))
    s[~mask] = 0
    han = signal.hanning(l)
    han2 = han[:, None] * han
    shan = han2 * s
    res, energies = find_close_function(shan, 8, beta=2, n_iter=100)
    np.save('null_space_eigvec_planewave.npy', res)
    #plt.imsave('null_space_eigvec_planewave.png', res, cmap='gray')

if im_nb == 2:
    thet1 = np.pi/16.
    thet2 = np.pi / 2 - np.pi/16.
    x, y = np.ogrid[:l, :l]
    mask = (x - l/2.)**2 + (y - l/2.)**2 < (l/2.)**2
    s1 = np.sin(np.pi/4 * (np.cos(thet1)*x  - np.sin(thet1)*y))
    s2 = np.sin(np.pi/4 * (np.cos(thet2)*x  - np.sin(thet2)*y))
    s = s1 + s2
    s[~mask] = 0
    han = signal.hanning(l)
    han2 = han[:, None] * han
    shan = han2 * s
    res, energies = find_close_function(shan, 8, beta=2, n_iter=120)
    np.save('null_space_eigvec.npy', res)
    # plt.imsave('null_space_eigvec.png', res, cmap='gray')


if im_nb == 3:
    thet1 = np.pi/8.
    thet2 = np.pi / 2 - np.pi/8.
    x, y = np.ogrid[:l, :l]
    mask = (x - l/2.)**2 + (y - l/2.)**2 < (l/2.)**2
    s1 = np.sin(np.pi/8 * (np.cos(thet1)*x  - np.sin(thet1)*y))
    s2 = np.sin(np.pi/8 * (np.cos(thet2)*x  - np.sin(thet2)*y))
    s = s1 + s2
    s[~mask] = 0
    han = signal.hanning(l)
    han2 = han[:, None] * han
    shan = han2 * s
    res, energies = find_close_function(shan, 4, beta=2, n_iter=120)

if im_nb == 4:
    im = generate_synthetic_data(l, n_pts=200)
    x, y = np.ogrid[:l, :l]
    mask = (x - l/2.)**2 + (y - l/2.)**2 < (l/2.)**2
    im[mask] -= im[mask].mean()
    res, energies = find_close_function(im, 4, beta=2, n_iter=80)
