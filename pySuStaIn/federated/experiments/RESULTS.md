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

- This validates the **ML fit** (federated EM). Federated MCMC uncertainty,
  federated number-of-subtypes selection (per-centre `f` + federated CVIC) and
  longitudinal handling are roadmap items (see README).
- The federated fit is currently **correct but slow** (Python per-centre loops +
  recomputation in the sequence search); inner-loop caching is the obvious next
  optimisation before large real-data runs.
- Harmonisation across centres is assumed upstream and out of scope.
