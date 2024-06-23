#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0.
# This file follows the PEP8 Python style guide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyclblast

# Settings for this sample
dtype = 'float16'
alpha = 1.5
alpha_fp16 = pyclblast.float32_to_float16(alpha)
n = 4

print("# Setting up OpenCL")
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("# Setting up Numpy arrays")
x = np.linspace(1.0, n, num=n).astype(dtype=dtype)
y = np.linspace(1.0, n / 2, num=n).astype(dtype=dtype)

print("# Setting up OpenCL arrays")
clx = Array(queue, x.shape, x.dtype)
cly = Array(queue, y.shape, y.dtype)
clx.set(x)
cly.set(y)

print("# Example level-1 operation: AXPY")
pyclblast.axpy(queue, n, clx, cly, alpha=alpha_fp16)
queue.finish()
print("# Result for vector y: %s" % cly.get())
print("# Expected result:     %s" % (alpha * x + y))
