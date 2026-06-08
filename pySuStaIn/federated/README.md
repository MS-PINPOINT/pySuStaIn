# Federated Z-score SuStaIn

A modular, **network-agnostic** federation layer for cross-sectional Z-score
SuStaIn. Centres keep their data local and return only **aggregate statistics**;
a server runs the *same* EM / greedy sequence search as pooled SuStaIn, driven
by those aggregates. There is (to our knowledge) no published federated SuStaIn,
so this is a new capability — the goal is to pool evidence across centres into a
**single, sharp** subtype+stage model instead of reconciling many noisy
per-centre fits.

## Why it's exact, not approximate

Everything SuStaIn reads from the data is the per-subject stage likelihood
`_calculate_likelihood_stage`, and every quantity the fit needs is a **sum over
subjects**:

- **log-likelihood:** `loglike = Σ_m log(total_prob_subj[m])`
- **f-update (subtype fractions):** `f_s ∝ Σ_m r[m,s]` (per-subject responsibilities)
- **sequence search:** each candidate ordering is scored by `Σ_m log(total_prob_subj[m])`

All model parameters (`stage_zscore`, `stage_biomarker_index`, min/max/std) come
from `Z_vals`/`Z_max` and are **data-independent** — identical at every centre.
So the server can reproduce pooled SuStaIn's EM exactly by aggregating per-centre
partial sums. With the same RNG and initial sequence, the federated EM makes
**identical** f-updates and sequence moves as pooled SuStaIn.

## What each centre returns (only aggregates — never raw data or per-subject values)

| Server need | Centre returns |
|---|---|
| f-update + convergence | `R_c[s] = Σ_m r[m,s]` (K-vector), `M_c`, `Σ_m log(total_prob_subj)` |
| sequence search | one `Σ_m log(total_prob_subj)` scalar per server-proposed candidate ordering |
| final assignment | per-subject subtype+stage computed **locally** (stays at the centre) |

The server proposes candidate sequences (data-independent generation, copied
verbatim from pooled `_optimise_parameters`) and only the likelihood evaluation
is federated.

## API

```python
from pySuStaIn.federated import ZscoreFederatedClient, FederatedZscoreSustain

clients = [ZscoreFederatedClient(centre_data, Z_vals, Z_max, labels, name=f"c{i}")
           for i, centre_data in enumerate(per_centre_data)]
fed = FederatedZscoreSustain(clients, Z_vals, Z_max, labels, N_S_max=2)

# one EM optimisation from a given init (matches pooled exactly):
S, f, loglike = fed.fit_em(S_init, f_init, rng)

# multi-start ML fit for a fixed number of subtypes:
S, f, loglike = fed.fit(N_S=2, n_startpoints=25, seed=0)

# per-centre ML subtype + stage (assignments computed locally):
assignments = fed.subtype_and_stage(S, f)
```

## Longitudinal data (interdependent visits per subject)

Cross-sectional SuStaIn treats every row as an independent subject. For
longitudinal data a **subject** has several visits that are *interdependent*:
they share one subtype and a **monotonically non-decreasing** SuStaIn stage.
This is implemented in `pySuStaIn/LongitudinalZscoreSustain.py` and changes only
the per-subject likelihood (a monotone-path dynamic program over the visits);
the EM, greedy sequence search and federation are inherited unchanged. With one
visit per subject it is **identical** to cross-sectional SuStaIn.

Data format: `visit_data` is `(n_visits × n_biomarkers)` positive Z-scores,
`subject_ids` `(n_visits,)` groups visits by subject, and a subject's rows are
assumed to be in **chronological order**.

```python
from pySuStaIn.LongitudinalZscoreSustain import LongitudinalZscoreSustain
from pySuStaIn.federated import LongitudinalFederatedClient, FederatedZscoreSustain

# pooled
model = LongitudinalZscoreSustain(visit_data, subject_ids, Z_vals, Z_max, labels,
                                  N_startpoints, N_S_max, N_iterations_MCMC,
                                  out, "ds", False)

# federated: one centre per site (visits of a subject stay together at its centre)
clients = [LongitudinalFederatedClient(vd_c, sid_c, Z_vals, Z_max, labels, name=f"c{i}")
           for i, (vd_c, sid_c) in enumerate(per_centre_visits)]
fed = FederatedZscoreSustain(clients, Z_vals, Z_max, labels, N_S_max=3)
S, f, loglike = fed.fit(N_S=3, n_startpoints=25, seed=0)
```

The federated layer is **model-agnostic** (`FederatedClient` wraps any model with
`_calculate_likelihood_stage` + `stage_zscore`), so cross-sectional and
longitudinal centres use the same server and aggregation — the longitudinal case
required no change to the federation code.

## Validation

`pySuStaIn/federated/experiments/run_validation.py` simulates ground-truth data
(`simulate.py` wraps the built-in Z-score simulator), splits subjects across
centres, and checks:

1. **Equivalence** — federated EM vs pooled EM from an identical init+RNG must
   match (sequences, fractions, log-likelihood).
2. **Recovery** — multi-start federated fit recovers the ground-truth sequences
   and agrees with an equivalent pooled multi-start fit.

```bash
python -m pySuStaIn.federated.experiments.run_validation \
    --n-biomarkers 10 --n-samples 1000 --n-subtypes 2 --n-centres 10 --startpoints 15
# add --heterogeneous to give centres uneven subtype prevalence
```

## Tests & CI

Two tiers, so users/CI don't pay for the expensive runs by default:

- **Cheap (run by default, in CI):** deterministic correctness — federated EM ==
  pooled EM, single-visit longitudinal == cross-sectional, federated-longitudinal
  == pooled-longitudinal.
  ```bash
  pytest tests/test_federated_zscore.py tests/test_longitudinal_zscore.py
  ```
- **Expensive (opt-in):** ground-truth recovery on simulated data (10 biomarkers,
  3 subtypes, multi-start). Skipped unless you ask for it:
  ```bash
  SUSTAIN_FULL=1 pytest tests/test_longitudinal_zscore.py -k recovers
  # or the full experiment with a report:
  python -m pySuStaIn.federated.experiments.run_validation_longitudinal \
      --n-biomarkers 10 --n-subtypes 3 --n-subjects 600 --n-visits 3 --n-centres 10
  ```

GitHub Actions (`.github/workflows/federated-tests.yml`) runs the **cheap** tests
on every push/PR with minimal deps (no `awkde`/`kde_ebm`). The **expensive**
recovery job is *not* run automatically — trigger it via the workflow's manual
“Run workflow” button (`run_expensive=true`) or run it locally as above.

## Scope & roadmap

- **Now:** cross-sectional **and longitudinal** Z-score models, ML fit via
  federated EM (exact), model-agnostic federation layer, candidate-pool
  initialisation, per-centre local assignment.
- **Not yet:** federated MCMC uncertainty (decomposes the same way — one scalar
  per centre per proposed sequence; communication-bound, batchable);
  federated number-of-subtypes selection via per-centre `f` + federated CVIC;
  inner-loop caching (current fit is correct but slow). Harmonisation is assumed
  upstream and out of scope here.
- **Communication:** the in-process experiments call methods directly; the
  client/server split is deliberately a clean boundary so it can be wired to an
  RPC / federated-orchestration backend without touching the algorithm.
