"""
In this script, we build a synthetic sinogram to be used with PyHST as well as
with the demo_tv_reconstruction.py script. This will help to compare the
different methods.
"""

import numpy as np
from reconstruction.util import generate_synthetic_data
from reconstruction.projections import build_projection_operator
from EdfFile import EdfFile

# Synthetic data
l = 256
x = generate_synthetic_data(l)


# Projection operator and projections data, with noise
H = build_projection_operator(l, l/5)
y = H * x.ravel()[:, np.newaxis]
y += 2*np.random.randn(*y.shape)
y[y<0] = 0

y = y.reshape((l/5, l))
y = y.astype(np.float32)

# Write EdfFile with float32 data
phantom = EdfFile("phantom_float.edf")

for line in y:
    lines = np.tile(line, (3, 1))
    print "writing", lines.shape
    phantom._WriteImage({}, lines, DataType="FloatValue", Append=1)
    #phantom.WriteImage({}, lines)

phantom._EdfFile__makeSureFileIsClosed()

y *= 2**15 / y.max()
y = y.astype(np.uint16)

# Write EdfFile with uint16 data
phantom = EdfFile("phantom.edf")

for line in y:
    lines = np.tile(line, (3, 1))
    phantom._WriteImage({}, lines, DataType="UnsignedShort", ByteOrder="LowByteFirst", Append=1)
    #phantom.WriteImage({}, lines)

phantom._EdfFile__makeSureFileIsClosed()
