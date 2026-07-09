"""Coverage-preserving synthetic data. Conformal coverage survives synthesis.
Verified: gaussian 90.6%, exponential 89.9%, bimodal 89.8%. Chad Edward Holland 2026.
Run: python3 synthesis.py"""
import numpy as np


class CoveragePreservingSynthesizer:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.coef_ = self.residuals_ = self.x_mean_ = self.x_std_ = None

    def fit(self, X, y):
        X = np.atleast_2d(X)
        if X.shape[0] != len(y): X = X.T
        self.x_mean_ = X.mean(axis=0); self.x_std_ = X.std(axis=0) + 1e-12
        Xs = (X - self.x_mean_) / self.x_std_
        A = np.column_stack([np.ones(len(y)), Xs])
        self.coef_, *_ = np.linalg.lstsq(A, y, rcond=None)
        self.residuals_ = y - A @ self.coef_
        return self

    def generate(self, n):
        Xs = self.rng.standard_normal((n, len(self.x_mean_)))
        X_synth = Xs * self.x_std_ + self.x_mean_
        A = np.column_stack([np.ones(n), Xs])
        return X_synth, A @ self.coef_ + self.rng.choice(self.residuals_, size=n, replace=True)


def conformal_threshold(true_vals, preds, confidence=0.9):
    scores = np.abs(np.asarray(true_vals) - np.asarray(preds))
    n = len(scores)
    return np.sort(scores)[min(int(np.ceil((n + 1) * confidence)), n) - 1]


def predict(X, coef, x_mean, x_std):
    Xs = (np.atleast_2d(X) - x_mean) / x_std
    return np.column_stack([np.ones(len(Xs)), Xs]) @ coef


def make_data(n, seed, noise="gaussian"):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3, 3, size=(n, 2))
    signal = 1.5 * X[:, 0] - 0.8 * X[:, 1] + 2.0
    if noise == "gaussian": eps = rng.normal(0, 1.2, n)
    elif noise == "exponential": eps = rng.exponential(1.2, n) - 1.2
    else: eps = np.where(rng.random(n) < 0.5, rng.normal(-2, 0.5, n), rng.normal(2, 0.5, n))
    return X, signal + eps


def coverage(y, preds, thr):
    return float(np.mean((y >= preds - thr) & (y <= preds + thr)))


if __name__ == "__main__":
    print("COVERAGE-PRESERVING SYNTHETIC DATA - falsifiable test\n" + "-" * 55)
    for noise in ["gaussian", "exponential", "bimodal"]:
        sc = []
        for t in range(30):
            Xr, yr = make_data(2000, t, noise)
            Xt, yt = make_data(3000, 1000 + t, noise)
            s = CoveragePreservingSynthesizer(t).fit(Xr, yr)
            pt = predict(Xt, s.coef_, s.x_mean_, s.x_std_)
            Xs, ys = s.generate(2000)
            ps = predict(Xs, s.coef_, s.x_mean_, s.x_std_)
            thr = conformal_threshold(ys, ps, 0.9)
            sc.append(coverage(yt, pt, thr))
        m = np.mean(sc)
        print(f"{noise:>12}: synth-calibrated coverage {m:.1%}  target 90%  {'PASS' if abs(m-0.9)<0.03 else 'FAIL'}")
    print("-" * 55 + "\nEvery number computed live. Vincit Omnia Veritas.")
