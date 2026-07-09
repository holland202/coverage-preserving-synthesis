# coverage-preserving-synthesis

**Synthetic data that keeps its statistical guarantee. Run the test yourself.**

Most synthetic data silently breaks downstream uncertainty estimates. If you release synthetic data and someone calibrates a conformal predictor on it, their "90% interval" often doesn't cover 90% on the real distribution — the guarantee fails quietly, and nobody notices until it matters.

This generator preserves that guarantee by construction, and ships a test that proves it.

```bash
python3 synthesis.py     # generator + falsifiable test, ~10s, numpy only
