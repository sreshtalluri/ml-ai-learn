# Adapted from AI Engineering from Scratch (MIT), phases/04-computer-vision/02-convolutions-from-scratch, phases/04-computer-vision/03-cnns-lenet-to-resnet
"""Convolution three ways: a multi-channel loop, the same thing as one matrix multiply (im2col),
and max pooling. Then the bookkeeping a CNN designer does by hand: output sizes, parameter
counts, receptive field, and why a residual block can start as the identity.

Run from guide/:  uv run code/13-deep-architectures/conv_scratch.py
"""
import numpy as np


# --- 1. Output size: how many window positions fit --------------------------------
def out_size(n, k, pad=0, stride=1):
    return (n + 2 * pad - k) // stride + 1


# --- 2. Multi-channel convolution: x [C_in, H, W], w [C_out, C_in, k, k] -----------
def conv2d(x, w, b=None, stride=1, pad=0):
    c_out, _, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = out_size(x.shape[1], k, 0, stride), out_size(x.shape[2], k, 0, stride)
    out = np.zeros((c_out, oh, ow))
    for o in range(c_out):                                  # one feature map per filter
        for i in range(oh):
            for j in range(ow):
                patch = x[:, i * stride:i * stride + k, j * stride:j * stride + k]
                out[o, i, j] = np.sum(patch * w[o])         # sums over channels too
    return out if b is None else out + b[:, None, None]


# --- 3. im2col: unroll every window into a column, then one matmul does it all -----
def conv2d_im2col(x, w, stride=1, pad=0):
    c_out, c_in, k, _ = w.shape
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    oh, ow = out_size(x.shape[1], k, 0, stride), out_size(x.shape[2], k, 0, stride)
    cols = np.stack([x[:, i * stride:i * stride + k, j * stride:j * stride + k].ravel()
                     for i in range(oh) for j in range(ow)], axis=1)   # [C_in*k*k, oh*ow]
    return (w.reshape(c_out, -1) @ cols).reshape(c_out, oh, ow)


# --- 4. Max pooling: keep the strongest response in each window --------------------
def max_pool(x, size=2):
    c, h, w = x.shape
    return x[:, :h // size * size, :w // size * size].reshape(c, h // size, size, w // size, size).max(axis=(2, 4))


# --- 5. Receptive field: how many input pixels one output pixel can see ------------
def receptive_field(layers):                                # layers: [(kernel, stride), ...]
    rf, jump = 1, 1
    for k, s in layers:
        rf += (k - 1) * jump                                # each layer widens by (k-1) input steps
        jump *= s                                           # stride spreads later layers out
    return rf


if __name__ == "__main__":
    print("== Course worked example 1: vertical-edge kernel on a 5x5 step image ==")
    img = np.zeros((1, 5, 5)); img[0, :, 3:] = 9            # columns 4-5 are 9
    edge = np.array([[[[-1, 0, 1]] * 3]], dtype=float)      # [1, 1, 3, 3]
    print(conv2d(img, edge)[0])
    print("sizes: no pad", out_size(5, 3), " pad 1", out_size(5, 3, 1), " stride 2", out_size(5, 3, 0, 2))

    print("\n== Course worked example 2: 2x2 max pool, stride 2 ==")
    x4 = np.array([[[1, 3, 2, 1], [4, 6, 5, 0], [1, 2, 9, 8], [3, 1, 4, 7]]], dtype=float)
    print(max_pool(x4)[0])

    print("\n== Loop vs im2col on random multi-channel data (synthetic) ==")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 8, 8)); w = rng.normal(size=(4, 3, 3, 3))
    a, b = conv2d(x, w, pad=1), conv2d_im2col(x, w, pad=1)
    print("output shape:", a.shape, " max difference:", f"{np.abs(a - b).max():.1e}")

    print("\n== Parameter count, worked example 3: 64 -> 128 channels, 3x3 ==")
    print("conv:", 128 * (64 * 3 * 3 + 1), " dense 32x32x64 -> 32x32x64:", f"{(32 * 32 * 64) ** 2:,}")

    print("\n== Receptive field ==")
    print("three 3x3 stride-1 convs:", receptive_field([(3, 1)] * 3), " (same view as one 7x7)")
    print("params, C=64 channels: three 3x3 =", 3 * 9 * 64 * 64, " one 7x7 =", 49 * 64 * 64)
    print("conv3 -> pool2 -> conv3 -> pool2 -> conv3:", receptive_field([(3, 1), (2, 2), (3, 1), (2, 2), (3, 1)]))

    print("\n== Residual block: y = x + F(x). With F's weights at 0, the block is the identity ==")
    relu = lambda z: np.maximum(0, z)
    x = relu(rng.normal(size=(4, 6, 6)))
    zero = np.zeros((4, 4, 3, 3))
    y = relu(x + conv2d(relu(conv2d(x, zero, pad=1)), zero, pad=1))
    print("output equals input:", bool(np.allclose(y, x)))
