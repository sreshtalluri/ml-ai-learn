# Adapted from AI Engineering from Scratch (MIT), phases/08-generative-ai/02-autoencoders-vae, phases/08-generative-ai/06-diffusion-ddpm-from-scratch
"""Generative models from scratch on synthetic data: a linear autoencoder trained by gradient
descent (and compared with PCA), the VAE's reparameterization trick and KL term, and a
complete 1-D DDPM: noise schedule, closed-form forward process, a noise predictor fitted to
the DDPM loss, and the reverse sampling loop.

Run from guide/:  uv run code/13-deep-architectures/generative_scratch.py
"""
import numpy as np

rng = np.random.default_rng(0)


# --- 1. Linear autoencoder: encode to k numbers, decode back, minimize squared error --
def train_autoencoder(X, k, lr=0.01, steps=3000):
    d = X.shape[1]
    E, D = rng.normal(scale=0.1, size=(d, k)), rng.normal(scale=0.1, size=(k, d))
    for _ in range(steps):
        Z = X @ E                                  # encode  [n, k]
        R = Z @ D - X                              # reconstruction error [n, d]
        gD = Z.T @ R / len(X)                      # dL/dD
        gE = X.T @ (R @ D.T) / len(X)              # dL/dE, chain rule through Z
        E, D = E - lr * gE, D - lr * gD
    return E, D


# --- 2. VAE pieces: sample z differentiably, and the KL penalty in closed form ---------
def reparameterize(mu, log_var, eps):
    return mu + np.exp(0.5 * log_var) * eps        # z = mu + sigma * eps; randomness is in eps


def kl_to_standard_normal(mu, log_var):
    return 0.5 * np.sum(np.exp(log_var) + mu ** 2 - 1 - log_var)


# --- 3. DDPM schedule and the closed-form forward process ------------------------------
T = 100
betas = np.linspace(1e-4, 0.2, T)                  # noise added at each step (larger than the
alphas = 1 - betas                                 # usual 0.02 because T is only 100)
alpha_bar = np.cumprod(alphas)                     # signal variance left after t steps


def noise(x0, t, eps):
    return np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps


# --- 4. Noise predictor: per-step least squares on a few nonlinear features -----------
def features(x):
    return np.stack([np.ones_like(x), x, np.tanh(x), np.tanh(2 * x), np.tanh(4 * x)], axis=-1)


def sample_data(n):                                # synthetic: two clusters at -2 and +2
    return rng.choice([-2.0, 2.0], size=n) + 0.3 * rng.normal(size=n)


def fit_noise_predictor(n=4000):
    W = np.zeros((T, 5))
    for t in range(T):                             # minimize ||eps - eps_hat(x_t, t)||^2
        x0, eps = sample_data(n), rng.normal(size=n)
        W[t] = np.linalg.lstsq(features(noise(x0, t, eps)), eps, rcond=None)[0]
    return W


# --- 5. Reverse process: start from pure noise, remove predicted noise step by step ---
def sample(W, n):
    x = rng.normal(size=n)                         # x_T ~ N(0, 1)
    for t in reversed(range(T)):
        eps_hat = features(x) @ W[t]
        x = (x - betas[t] / np.sqrt(1 - alpha_bar[t]) * eps_hat) / np.sqrt(alphas[t])
        if t > 0:
            x += np.sqrt(betas[t]) * rng.normal(size=n)   # fresh noise, except at the last step
    return x


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("== Linear autoencoder, synthetic 5-d data that lies near a 2-d plane ==")
    X = rng.normal(size=(500, 2)) @ rng.normal(size=(2, 5)) + 0.05 * rng.normal(size=(500, 5))
    X -= X.mean(axis=0)
    E, D = train_autoencoder(X, k=2)
    ae_mse = np.mean((X @ E @ D - X) ** 2)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pca_mse = np.mean((X @ Vt[:2].T @ Vt[:2] - X) ** 2)
    print(f"reconstruction MSE: autoencoder {ae_mse:.5f}, PCA {pca_mse:.5f}, data variance {X.var():.3f}")

    print("\n== VAE reparameterization and KL ==")
    mu, log_var = np.array([1.0, 0.0]), np.array([0.0, np.log(0.25)])
    print("z for eps = [0.5, -2]:", reparameterize(mu, log_var, np.array([0.5, -2.0])))
    print("KL(N(mu, sigma^2) || N(0, 1)):", round(kl_to_standard_normal(mu, log_var), 4))
    print("KL when mu = 0, sigma = 1:", kl_to_standard_normal(np.zeros(2), np.zeros(2)))

    print("\n== Course worked example: x0 = 2.0, alpha_bar = 0.64, eps = -0.5 ==")
    x_t = np.sqrt(0.64) * 2.0 + np.sqrt(0.36) * -0.5
    print(f"x_t = {x_t:.2f};  recovered x0 = {(x_t - np.sqrt(0.36) * -0.5) / np.sqrt(0.64):.2f}")

    print(f"\n== DDPM schedule, T = {T} ==")
    for t in (0, 9, 49, 99):
        print(f"t = {t + 1:3d}: alpha_bar = {alpha_bar[t]:.4f}  (signal std {np.sqrt(alpha_bar[t]):.3f})")

    W = fit_noise_predictor()
    x = sample(W, 4000)
    print("\n== Samples from the trained 1-D DDPM (data: clusters at -2 and +2) ==")
    print(f"fraction near -2: {np.mean(np.abs(x + 2) < 1):.2f}, near +2: {np.mean(np.abs(x - 2) < 1):.2f}, "
          f"between (|x| < 1): {np.mean(np.abs(x) < 1):.2f}")
    print(f"mean |x| = {np.abs(x).mean():.2f} (data: about 2.0)")
