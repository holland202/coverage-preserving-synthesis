# coverage-preserving-synthesis

**Synthetic data that keeps its statistical guarantee. Run the test yourself.**

Most synthetic data silently breaks downstream uncertainty estimates. If you
release synthetic data and someone calibrates a conformal predictor on it, their
"90% interval" often does not cover 90% on the real distribution -- the
guarantee fails quietly, and nobody notices until it matters.

This generator preserves that guarantee by construction, and ships a test that
proves it.

    python3 synthesis.py     # generator + falsifiable test, ~10s, numpy only

Author: Chad Edward Holland. Runs on a phone (Python 3, numpy). 2026.

## The claim, stated so it can be falsified

A 90% conformal prediction interval calibrated on SYNTHETIC data from this
generator still covers ~90% of the truth on the REAL distribution -- even when
the real noise is non-Gaussian.

Verified on-device, mean of 30 trials each:

    real noise     synthetic-calibrated coverage   target
    Gaussian       90.6%                            90%
    Exponential    89.9%                            90%
    Bimodal        89.8%                            90%

The bimodal case is the real test: naive "add Gaussian noise" synthesis fails
there, because it cannot reproduce a two-cluster residual distribution. This
generator resamples the REAL empirical residuals, so the synthetic nonconformity
distribution matches -- and coverage transfers.

## How it works

Conformal coverage depends only on the distribution of nonconformity scores
|y - f(x)|. So the object that must be preserved is the residual distribution,
not the marginal or the trend. The generator fits a linear trend, keeps the
empirical residuals, and generates synthetic samples by bootstrapping those real
residuals onto synthetic covariates. By Glivenko-Cantelli the synthetic residual
law converges to the real one, so coverage transfers. Full derivation in
MATHEMATICS.md.

## Honest scope

This is a correct, verifiable composition of established methods -- split
conformal prediction (Vovk et al.) and the residual bootstrap (Efron). It is NOT
a novel algorithm. Its value is a specific, falsifiable guarantee about synthetic
data, proven by execution, in ~60 lines of numpy that run on a phone.

It extends the primitives in github.com/holland202/edge-ai-primitives from
prediction to synthesis.

Not guaranteed under: distribution shift between calibration and test data;
heteroscedastic residual laws beyond what the pooled residual sample captures;
cases where the Gaussian covariate approximation distorts which residuals
co-occur with which covariates. These limits are real and stated here, not hidden.

Vincit Omnia Veritas -- truth conquers, but only when checked.
