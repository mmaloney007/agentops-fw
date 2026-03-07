#!/usr/bin/env python3
"""Generate a simple 2-input add model to test multi-input bootstrap."""
import warnings
warnings.filterwarnings("ignore")
import torch  # noqa: F401 — must import before coremltools
import numpy as np
import coremltools as ct
from coremltools.converters.mil.mil import Builder as mb

D, S = 64, 16

@mb.program(input_specs=[
    mb.TensorSpec(shape=(1, D, 1, S)),
    mb.TensorSpec(shape=(1, D, 1, S)),
])
def prog(a, b):
    a16 = mb.cast(x=a, dtype="fp16", name="a16")
    b16 = mb.cast(x=b, dtype="fp16", name="b16")
    s = mb.add(x=a16, y=b16, name="sum16")
    out = mb.cast(x=s, dtype="fp32", name="out")
    return out

model = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE,
                   minimum_deployment_target=ct.target.macOS15)
model.save("/tmp/test_multi_input.mlpackage")
print("Saved /tmp/test_multi_input.mlpackage")
