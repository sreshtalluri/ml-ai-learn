# Adapted from AI Engineering from Scratch (MIT), phases/01-math-foundations/01-linear-algebra-intuition, 02-vectors-matrices-operations, 03-matrix-transformations
"""Vectors and matrices in plain Python (lists only), then checked against NumPy.

Run from guide/:  uv run code/00-foundations/vectors_scratch.py
"""
import math

import numpy as np


# --- vector operations -------------------------------------------------------
def dot(a, b):
    assert len(a) == len(b), f"shape mismatch: {len(a)} vs {len(b)}"
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def project(c, a):
    """The part of c that points along a: (c·a / a·a) a."""
    s = dot(c, a) / dot(a, a)
    return [s * ai for ai in a]


# --- matrix operations -------------------------------------------------------
def shape(M):
    return (len(M), len(M[0]))


def transpose(M):
    return [list(col) for col in zip(*M)]


def matmul(A, B):
    (n, k), (k2, m) = shape(A), shape(B)
    assert k == k2, f"inner dims differ: {shape(A)} @ {shape(B)}"
    Bt = transpose(B)                                   # columns of B as rows
    return [[dot(row, col) for col in Bt] for row in A]  # entry (i, j) = row i · column j


# --- a matrix is a transformation ---------------------------------------------
def rotation(theta):
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s], [s, c]]


def apply(M, v):
    return [dot(row, v) for row in M]


if __name__ == "__main__":
    a, c = [3, 1], [1, 2]
    print(f"a·c = {dot(a, c)},  |a| = {norm(a):.4f},  cos θ = {cosine(a, c):.4f}")
    print(f"projection of c on a = {[round(p, 2) for p in project(c, a)]}")

    x, w, b = [2, 1.5], [30, 120], 50
    print(f"house price x·w + b = {dot(x, w) + b}")

    A = [[1, 2], [3, 4], [5, 6]]
    B = [[1, 0, 2], [0, 1, 1]]
    AB = matmul(A, B)
    print(f"A {shape(A)} @ B {shape(B)} = {shape(AB)}: {AB}")
    try:
        matmul(A, A)
    except AssertionError as e:
        print(f"A @ A -> {e}")

    R = rotation(math.pi / 2)
    print(f"rotate (1, 0) by 90°: {[round(v, 6) for v in apply(R, [1, 0])]}")
    R2 = matmul(R, R)
    print(f"two 90° rotations = one 180°: {[round(v, 6) for v in apply(R2, [1, 0])]}")

    # Check against NumPy
    assert np.isclose(dot(a, c), np.dot(a, c))
    assert np.isclose(cosine(a, c), np.dot(a, c) / np.linalg.norm(a) / np.linalg.norm(c))
    assert np.allclose(AB, np.array(A) @ np.array(B))
    assert np.allclose(apply(R, [1, 0]), [0, 1])
    print("all checks match NumPy")
