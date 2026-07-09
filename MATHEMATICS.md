# Coverage-Preserving Synthetic Data — Mathematical Basis

Author: Chad Edward Holland, 2026. Verified by execution.

## Problem

Real data: (X, Y) ~ P, Y = f(X) + e, e ~ D (residual law). Goal: a synthetic
distribution Q such that a split conformal predictor calibrated on synthetic
data from Q still covers on the REAL P:

    P( Y in C_synth(X) ) >= 1 - alpha.

## Why naive synthesis fails

Split conformal sets the interval half-width to the empirical (1-alpha)-quantile
of nonconformity scores s = |Y - fhat(X)|. Coverage on P depends ONLY on the
distribution of s under P. If synthetic residuals differ from real ones,
q_synth != q_real, and coverage breaks. Preserving the marginal or the trend is
NOT enough -- the conditional residual distribution is what must be preserved.

## Construction

1. Fit fhat by least squares on standardized covariates.
2. Retain empirical residuals R = { y_i - fhat(x_i) } (nonparametric est. of D).
3. Generate: X' ~ N(mu, sigma^2); e' ~ Uniform(R); Y' = fhat(X') + e'.

By Glivenko-Cantelli, resampling R gives D' -> D in distribution as |R| grows,
so q_synth -> q_real in probability, so coverage on P -> 1 - alpha.

## Scope

VALID under exchangeability; residual law capturable by resampling; trend
mis-specification absorbed into R (this is why it is robust for bimodal noise).
NOT guaranteed under covariate shift interacting with heteroscedastic residuals,
or where the Gaussian covariate approximation distorts residual-covariate pairing.

## Verification (synthesis.py)

Target 0.90. Synthetic-calibrated coverage on real test data, mean of 30 trials:
Gaussian 90.6%, Exponential 89.9%, Bimodal 89.8%. All within 3 points.

## Positioning

Correct, verifiable composition of established methods (Vovk conformal; Efron
bootstrap). Not novel. Value: a falsifiable synthetic-data guarantee, proven by
execution, numpy-only, on a phone.
