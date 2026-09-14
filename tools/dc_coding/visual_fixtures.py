"""Deterministic float diagnostic ramps and explicit sRGB transfer functions.

Explicit ufunc outputs avoid operand reuse observed for large float64
temporaries with NumPy 2.2.6 under the installed Python 3.14 runtime.
"""
import numpy as np


def linear(values):
    x = np.asarray(values)
    result = np.empty_like(x)
    low = x <= .04045
    np.divide(x, 12.92, out=result, where=low)
    high = np.logical_not(low)
    np.add(x, .055, out=result, where=high)
    np.divide(result, 1.055, out=result, where=high)
    np.power(result, 2.4, out=result, where=high)
    return result


def srgb(values):
    x = np.maximum(values, 0)
    result = np.empty_like(x)
    low = x <= .0031308
    np.multiply(x, 12.92, out=result, where=low)
    high = np.logical_not(low)
    np.power(x, 1 / 2.4, out=result, where=high)
    np.multiply(result, 1.055, out=result, where=high)
    np.subtract(result, .055, out=result, where=high)
    return result


def make_gradients():
    x, y = np.meshgrid(np.linspace(0, 1, 2048), np.linspace(0, 1, 1024))
    t, vertical = np.empty_like(x), np.empty_like(y)
    np.multiply(x, .85, out=t)
    np.square(y, out=vertical)
    np.multiply(vertical, .15, out=vertical)
    np.add(t, vertical, out=t)
    assert float(t.min()) == 0 and float(t.max()) == 1
    signals = {}
    for name, base, amplitude in [("gray-gradient", .22, .12), ("dark-gradient", .015, .105)]:
        ramp = np.empty_like(t)
        np.multiply(t, amplitude, out=ramp)
        np.add(ramp, base, out=ramp)
        signals[name] = linear(np.repeat(ramp[:, :, None], 3, axis=2))
    sky = np.empty((*t.shape, 3))
    for channel, base, amplitude in [(0, .32, .23), (1, .48, .20), (2, .68, .14)]:
        np.multiply(t, amplitude, out=sky[:, :, channel])
        np.add(sky[:, :, channel], base, out=sky[:, :, channel])
    signals["sky-gradient"] = linear(sky)
    signed = np.empty_like(t)
    np.multiply(t, 2, out=signed)
    np.subtract(signed, 1, out=signed)
    chroma = np.empty_like(sky)
    np.multiply(signed, .04, out=chroma[:, :, 0])
    np.add(chroma[:, :, 0], .18, out=chroma[:, :, 0])
    np.multiply(signed, -(.2126 / .7152) * .04, out=chroma[:, :, 1])
    np.add(chroma[:, :, 1], .18, out=chroma[:, :, 1])
    chroma[:, :, 2].fill(.18)
    signals["chroma-gradient"] = chroma
    return signals
