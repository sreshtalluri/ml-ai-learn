# Adapted from AI Engineering from Scratch (MIT), phases/08-generative-ai/07-latent-diffusion-stable-diffusion
"""Latent diffusion in miniature, in NumPy: a frozen linear autoencoder (stage 1) plus a
class-conditional DDPM with classifier-free guidance that runs only in its 2-D latent space (stage 2).

Data is SYNTHETIC: 16-D "images" that lie near a 2-D plane, in two classes.
Fixed seed, CPU only, runs in a few seconds.
Run from guide/:  uv run code/13-deep-architectures/latent_diffusion_scratch.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import BLUE, GRAY, ORANGE, PURPLE, plt, save  # noqa: E402

rng = np.random.default_rng(0)
D_PIX, D_LAT, T, NULL = 16, 2, 50, 2          # data dim, latent dim, diffusion steps, null-class id

# ---------------------------------------------------------------------------
# SYNTHETIC data: two classes on a 2-D plane embedded in 16-D, plus small noise
# ---------------------------------------------------------------------------
BASIS = np.linalg.qr(rng.standard_normal((D_PIX, D_LAT)))[0].T * 6.0   # [2, 16]
CENTERS = np.array([[-1.0, 0.5], [1.0, -0.5]])                          # class centres in the plane


def sample_data(n):
    c = rng.integers(0, 2, n)
    plane = CENTERS[c] + 0.35 * rng.standard_normal((n, D_LAT))
    return plane @ BASIS + 0.05 * rng.standard_normal((n, D_PIX)), c    # [n, 16], [n]


# ---------------------------------------------------------------------------
# Stage 1: the autoencoder. Here PCA (a linear autoencoder), fitted once and frozen.
# ---------------------------------------------------------------------------
X_fit, _ = sample_data(5000)
MU = X_fit.mean(0)
ENC = np.linalg.svd(X_fit - MU, full_matrices=False)[2][:D_LAT].T     # [16, 2] top principal directions
Z_raw = (X_fit - MU) @ ENC
SCALE = 1.0 / Z_raw.std()                                              # like SD's VAE scaling factor


def encode(x):
    return (x - MU) @ ENC * SCALE          # [n, 16] -> [n, 2], roughly unit variance


def decode(z):
    return (z / SCALE) @ ENC.T + MU        # [n, 2] -> [n, 16]


# ---------------------------------------------------------------------------
# Stage 2: an epsilon-prediction MLP eps(z_t, t, c), trained with label dropout
# ---------------------------------------------------------------------------
betas = np.linspace(1e-4, 0.2, T)          # short schedule: T = 50 steps reach near-pure noise
alphas = 1 - betas
abar = np.cumprod(alphas)


def features(z, t, c):
    tt = t / T
    temb = np.stack([np.sin(np.pi * tt), np.cos(np.pi * tt), np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt)], 1)
    return np.hstack([z, temb, np.eye(3)[c]])                 # [n, 2 + 4 + 3]


class MLP:
    def __init__(self, d_in, d_out, h=128):
        s = lambda a, b: rng.standard_normal((a, b)) / np.sqrt(a)  # noqa: E731
        self.p = {"W1": s(d_in, h), "b1": np.zeros(h), "W2": s(h, h), "b2": np.zeros(h),
                  "W3": s(h, d_out), "b3": np.zeros(d_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X):
        p = self.p
        h1 = np.tanh(X @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        self.cache = (X, h1, h2)
        return h2 @ p["W3"] + p["b3"]

    def backward(self, d_out):
        X, h1, h2 = self.cache
        p = self.p
        g = {"W3": h2.T @ d_out, "b3": d_out.sum(0)}
        d2 = (d_out @ p["W3"].T) * (1 - h2**2)
        g["W2"], g["b2"] = h1.T @ d2, d2.sum(0)
        d1 = (d2 @ p["W2"].T) * (1 - h1**2)
        g["W1"], g["b1"] = X.T @ d1, d1.sum(0)
        return g

    def adam(self, g, lr=2e-3, b1=0.9, b2=0.999):
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh, vh = self.m[k] / (1 - b1**self.t), self.v[k] / (1 - b2**self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def train_step(net, batch=256, p_uncond=0.1):
    x0, c = sample_data(batch)
    z0 = encode(x0)                                      # diffusion never sees the 16-D data
    t = rng.integers(0, T, batch)
    eps = rng.standard_normal(z0.shape)
    zt = np.sqrt(abar[t])[:, None] * z0 + np.sqrt(1 - abar[t])[:, None] * eps
    c = np.where(rng.random(batch) < p_uncond, NULL, c)   # label dropout: learn eps_uncond too
    pred = net.forward(features(zt, t, c))
    net.adam(net.backward(2 * (pred - eps) / pred.size))
    return np.mean((pred - eps) ** 2)


def sample(net, c, s, n):
    """DDPM ancestral sampling with classifier-free guidance scale s (s = 1: plain conditional)."""
    z = rng.standard_normal((n, D_LAT))
    cc, cu = np.full(n, c), np.full(n, NULL)
    for t in range(T - 1, -1, -1):
        tt = np.full(n, t)
        e_c = net.forward(features(z, tt, cc))
        e_u = net.forward(features(z, tt, cu))
        e = e_u + s * (e_c - e_u)                        # CFG
        z = (z - betas[t] / np.sqrt(1 - abar[t]) * e) / np.sqrt(alphas[t])
        if t > 0:
            z += np.sqrt(betas[t]) * rng.standard_normal(z.shape)
    return z


def main():
    print("== Compression arithmetic (Stable Diffusion family) ==")
    for name, (h, w, ch), (lh, lw, lc) in [("SD 1.5, 512px", (512, 512, 3), (64, 64, 4)),
                                           ("SDXL, 1024px", (1024, 1024, 3), (128, 128, 4)),
                                           ("SD3 / Flux, 1024px", (1024, 1024, 3), (128, 128, 16))]:
        px, lat = h * w * ch, lh * lw * lc
        print(f"{name:>18}: {px:>9,} pixel values -> {lat:>7,} latent values = {px / lat:.0f}x fewer")

    print("\n== Stage 1: linear autoencoder on SYNTHETIC 16-D data ==")
    x_test, c_test = sample_data(2000)
    rec = decode(encode(x_test))
    print(f"latent std before scaling: {Z_raw.std():.3f}, scale factor: {SCALE:.4f}, after: {encode(X_fit).std():.3f}")
    print(f"reconstruction MSE: {np.mean((rec - x_test) ** 2):.4f}  (data variance {x_test.var(0).sum() / D_PIX:.4f})")

    print("\n== Stage 2: train the latent denoiser ==")
    net = MLP(D_LAT + 4 + 3, D_LAT)
    for step in range(1, 4001):
        loss = train_step(net)
        if step in (1, 1000, 4000):
            print(f"step {step:>4}: noise-prediction MSE {loss:.3f}")

    print("\n== Worked CFG arithmetic ==")
    e_u, e_c, s = 0.2, 0.5, 3.0
    print(f"eps_uncond = {e_u}, eps_cond = {e_c}, s = {s}: eps = {e_u} + {s} * ({e_c} - {e_u}) = {e_u + s * (e_c - e_u):.2f}")

    print("\n== Guidance sweep (class 1; true class centre in latent space shown below) ==")
    true_c1 = encode(CENTERS[[1]] @ BASIS)[0]
    z_real = encode(x_test[c_test == 1])
    print(f"real class-1 latents: mean {np.round(z_real.mean(0), 2)}, spread {z_real.std(0).mean():.2f}")
    print(f"{'scale s':>7} | {'mean latent':>15} | {'spread':>6} | {'dist to centre':>14}")
    sweep = {}
    for s in [0.0, 1.0, 3.0, 7.0]:
        z = sample(net, 1, s, 1000)
        sweep[s] = z
        print(f"{s:>7.1f} | {str(np.round(z.mean(0), 2)):>15} | {z.std(0).mean():>6.2f} | "
              f"{np.linalg.norm(z.mean(0) - true_c1):>14.2f}")
    x_gen = decode(sweep[1.0])
    print(f"decoded samples (s = 1): latent {sweep[1.0].shape} -> data {x_gen.shape}")

    # ---------------------------- figure ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    z_all = encode(x_test)
    for k, col in [(0, GRAY), (1, BLUE)]:
        m = c_test == k
        ax[0].scatter(z_all[m, 0], z_all[m, 1], s=4, color=col, alpha=0.4, label=f"real class {k}")
    ax[0].scatter(sweep[0.0][:400, 0], sweep[0.0][:400, 1], s=4, color=PURPLE, alpha=0.6, label="s = 0 (unconditional)")
    ax[0].set(title="Latent space: real data and unguided samples", xlabel="z1", ylabel="z2")
    ax[0].legend(fontsize=8, markerscale=3)
    m = c_test == 1
    ax[1].scatter(z_all[m, 0], z_all[m, 1], s=4, color=GRAY, alpha=0.3, label="real class 1")
    for s, col in [(1.0, BLUE), (7.0, ORANGE)]:
        ax[1].scatter(sweep[s][:400, 0], sweep[s][:400, 1], s=4, color=col, alpha=0.6, label=f"class 1, s = {s:g}")
    ax[1].set(title="Guidance: s = 7 is tighter but off-target", xlabel="z1", ylabel="z2")
    ax[1].legend(fontsize=8, markerscale=3)
    save(fig, "latent-diffusion-stable-diffusion")


if __name__ == "__main__":
    main()
