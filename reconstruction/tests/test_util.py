import numpy as np
from ..util import generate_synthetic_data

def test_generate_data():
    l_x = 64
    data = generate_synthetic_data(l_x)
    assert data.shape == (l_x, l_x)
    assert np.all(np.unique(data) == [0, 1])
    x, y = np.ogrid[:l_x, :l_x]
    mask_outer = (x - l_x / 2) ** 2 + (y - l_x / 2) ** 2 > (l_x / 2) ** 2
    assert np.all(data[mask_outer] == 0)
    # Do not crop outside the central circle
    data = generate_synthetic_data(l_x, crop=False)
    assert data[mask_outer].sum() > 0
