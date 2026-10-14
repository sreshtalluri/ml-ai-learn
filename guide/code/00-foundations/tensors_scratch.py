# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/12-tensor-operations
"""How NumPy arrays work underneath: flat storage + shape + strides, reshape and transpose as views,
the broadcasting rule, and why vectorized code is faster. Data is SYNTHETIC.

Run from guide/:  uv run code/00-foundations/tensors_scratch.py
"""
import time

import numpy as np


def strides_for(shape):
    """Row-major (C order): the last axis moves fastest, so its stride is 1 element."""
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def flat_index(index, strides):
    return sum(i * s for i, s in zip(index, strides))


def broadcast_shape(a, b):
    """Align shapes from the right; each pair of dims must be equal or one of them must be 1."""
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da != db and 1 not in (da, db):
            raise ValueError(f"cannot broadcast {a} with {b}: {da} vs {db}")
        out.append(max(da, db))
    return tuple(reversed(out))


if __name__ == "__main__":
    data = list(range(24))                       # one flat buffer
    shape = (2, 3, 4)
    st = strides_for(shape)
    print(f"shape {shape} -> strides {st} (elements)")
    print(f"element [1, 2, 3] lives at flat position {flat_index((1, 2, 3), st)} -> {data[flat_index((1, 2, 3), st)]}")

    t = np.arange(24).reshape(shape)
    print(f"NumPy strides in bytes {t.strides} = {st} × {t.itemsize} bytes")
    tt = t.transpose(2, 0, 1)
    print(f"transpose is a view: shape {tt.shape}, strides {tt.strides}, shares memory: {np.shares_memory(t, tt)}")
    print(f"contiguous after transpose? {tt.flags['C_CONTIGUOUS']}")

    for a, b in [((1000, 3), (3,)), ((4, 1), (1, 5)), ((8, 1, 6), (7, 1)), ((3, 2), (3,))]:
        try:
            print(f"broadcast {a} with {b} -> {broadcast_shape(a, b)}")
        except ValueError as e:
            print(e)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(200_000, 3))
    t0 = time.perf_counter()
    loop = [[(X[i, j] - 0.5) * 2 for j in range(3)] for i in range(X.shape[0])]
    t1 = time.perf_counter()
    vec = (X - 0.5) * 2
    t2 = time.perf_counter()
    print(f"python loop {t1 - t0:.3f}s, vectorized {t2 - t1:.4f}s, same result: {np.allclose(loop, vec)}")

    assert st == [s // t.itemsize for s in t.strides]
    assert broadcast_shape((8, 1, 6), (7, 1)) == np.broadcast_shapes((8, 1, 6), (7, 1))
    assert broadcast_shape((4, 1), (1, 5)) == (4, 5)
    print("all checks match NumPy")
