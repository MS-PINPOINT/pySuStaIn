# Federated vs pooled Z-score SuStaIn — validation results

Federation simulated in-process across N centres (each returns aggregates only).
Reproduce with `run_validation.py` (see README).

## Target config — 1000 subjects, 10 biomarkers, 2 subtypes, 10 centres, 15 start-points

```
=== 1. EQUIVALENCE: federated EM vs pooled EM (same init + RNG) ===
  pooled loglike = -16506.811831
  fed    loglike = -16506.811831
  |loglike diff| = 0.000e+00
  |f diff|       = 1.110e-16
  sequences identical = True (max element diff 0)

=== 2. RECOVERY: multi-start fit (N_S=2, 15 starts) vs ground truth ===
  fed    loglike = -16367.1296   fractions = [0.582 0.418]   (ground truth 0.6/0.4)
  pooled loglike = -16367.1296   fractions = [0.582 0.418]
  Kendall tau  federated vs ground truth : [0.963 0.954]  (mean 0.959)
  Kendall tau  pooled    vs ground truth : [0.963 0.954]  (mean 0.959)
  Kendall tau  federated vs pooled       : [1.000 1.000]  (mean 1.000)
```

## Longitudinal (interdependent visits per subject)

Subjects have multiple visits sharing one subtype and a monotonically
non-decreasing stage (`LongitudinalZscoreSustain`). Reproduce with
`run_validation_longitudinal.py`.

Full-scale equivalence — 600 subjects × 3 visits, 10 biomarkers, **3 subtypes**,
10 centres, 15 starts:

```
=== EQUIVALENCE: federated-long EM vs pooled-long EM (same init + RNG) ===
  pooled loglike = -29448.392220
  fed    loglike = -29448.392220
  |loglike diff| = 7.276e-12     |f diff| = 1.665e-16
  sequences identical = True
```

Recovery — 150 subjects × 3 visits, 5 biomarkers, 2 subtypes, 5 centres:

```
  fed    loglike = -3872.983     fractions = [0.557 0.443]
  Kendall tau  federated vs ground truth : [0.924 0.905]  (mean 0.914)
  Kendall tau  federated vs pooled       : [1.000 1.000]  (mean 1.000)
```

Sanity: with **one visit per subject** the longitudinal model is *identical* to
cross-sectional (`_calculate_likelihood_stage` diff `0`, pinned by
`tests/test_regression_golden.py` and `tests/test_longitudinal_zscore.py`). The
opt-in `SUSTAIN_FULL=1` test runs the full 10-biomarker / 3-subtype recovery.

## Takeaways

- **Federation is exact.** From an identical init + RNG, federated EM reproduces
  pooled EM to machine precision (log-likelihood diff `0`, identical sequences,
  `f` diff ~1e-16). Confirmed at 200×5 and 1000×10 (10 centres).
- **No accuracy cost.** The multi-start federated fit is indistinguishable from
  pooled (identical log-like and fractions; Kendall τ fed-vs-pooled = 1.000) and
  recovers the ground-truth subtype sequences at mean τ ≈ 0.96 at a manageable
  per-centre N (100 subjects/centre).
- **Aggregates only.** Centres return responsibility sums, log-like scalars and
  per-candidate sequence scores — never raw data or per-subject values.

## Notes / limitations

- This validates the **ML fit** (federated EM), cross-sectional **and
  longitudinal**. Federated MCMC uncertainty and federated number-of-subtypes
  selection (per-centre `f` + federated CVIC) remain roadmap items (see README).
- The hot paths are now optimised (bounded per-sequence stage-likelihood cache;
  vectorised longitudinal monotone-path DP), verified numerically exact by the
  golden regression tests. Further speed-ups (e.g. batching the sequence search)
  are possible before very large real-data runs.
- Harmonisation across centres is assumed upstream and out of scope.
