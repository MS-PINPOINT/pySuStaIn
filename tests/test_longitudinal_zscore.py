"""Longitudinal Z-score SuStaIn: cheap correctness checks + an opt-in expensive
recovery test.

Cheap (always run in CI):
  * single-visit longitudinal == cross-sectional (exact)
  * federated-longitudinal EM == pooled-longitudinal EM (same init + RNG)

Expensive (opt-in only; set SUSTAIN_FULL=1 to run):
  * ground-truth recovery, 10 biomarkers, 3 subtypes, multi-start.
The expensive test is skipped by default because it is slow; run it locally or
via the manual CI dispatch.
"""
import os
import tempfile

import numpy as np
import pytest

from pySuStaIn.ZscoreSustain import ZscoreSustain
from pySuStaIn.LongitudinalZscoreSustain import LongitudinalZscoreSustain
from pySuStaIn.federated.client import LongitudinalFederatedClient
from pySuStaIn.federated.server import FederatedZscoreSustain
from pySuStaIn.federated.simulate import (
    simulate_zscore, simulate_longitudinal, split_subjects_into_centres,
)
from pySuStaIn.federated.experiments.run_validation import seq_tau, align_subtypes

RUN_EXPENSIVE = os.environ.get("SUSTAIN_FULL") == "1"


def _long(visit_data, subject_ids, Zv, Zm, labels, N_S_max=2):
    return LongitudinalZscoreSustain(
        visit_data, subject_ids, Zv, Zm, labels, 1, N_S_max, 1,
        tempfile.mkdtemp(prefix="long_test_"), "l", False, 0)


def test_single_visit_reduces_to_cross_sectional():
    sim = simulate_zscore(n_biomarkers=5, n_samples=120, n_subtypes=2, seed=7)
    data, Zv, Zm, labels = sim["data"], sim["Z_vals"], sim["Z_max"], sim["labels"]
    xs = ZscoreSustain(data, Zv, Zm, labels, 1, 1, 1, tempfile.mkdtemp(), "x", False, 0)
    lg = _long(data, np.arange(data.shape[0]), Zv, Zm, labels, N_S_max=1)
    S = xs._initialise_sequence(xs._AbstractSustain__sustainData, np.random.default_rng(1))[0]
    p_xs = xs._calculate_likelihood_stage(xs._AbstractSustain__sustainData, S)
    p_lg = lg._calculate_likelihood_stage(lg._AbstractSustain__sustainData, S)
    assert np.max(np.abs(p_xs - p_lg)) < 1e-12


def test_federated_longitudinal_equals_pooled():
    sim = simulate_longitudinal(n_biomarkers=5, n_subjects=120, n_subtypes=2,
                                n_visits=3, seed=4)
    vd, sid, Zv, Zm, labels = (sim["visit_data"], sim["subject_ids"], sim["Z_vals"],
                               sim["Z_max"], sim["labels"])
    pooled = _long(vd, sid, Zv, Zm, labels, N_S_max=2)
    psd = pooled._AbstractSustain__sustainData
    rows = split_subjects_into_centres(sid, n_centres=4, seed=4)
    clients = [LongitudinalFederatedClient(vd[r], sid[r], Zv, Zm, labels, name=f"c{i}")
               for i, r in enumerate(rows)]
    fed = FederatedZscoreSustain(clients, Zv, Zm, labels, N_S_max=2)

    S0 = np.array([pooled._initialise_sequence(psd, np.random.default_rng(4))[0] for _ in range(2)])
    f0 = np.ones(2) / 2
    ps, pf, pl, *_ = pooled._perform_em(psd, S0.copy(), f0.copy(), np.random.default_rng(9))
    fs, ff, fl = fed.fit_em(S0.copy(), f0.copy(), np.random.default_rng(9))
    assert abs(pl - fl) < 1e-6
    assert np.array_equal(ps.astype(int), fs.astype(int))
    assert np.allclose(pf, ff, atol=1e-10)


@pytest.mark.skipif(not RUN_EXPENSIVE, reason="expensive; set SUSTAIN_FULL=1 to run")
def test_longitudinal_recovers_ground_truth():
    sim = simulate_longitudinal(n_biomarkers=10, n_subjects=600, n_subtypes=3,
                                n_visits=3, seed=42)
    vd, sid, Zv, Zm, labels = (sim["visit_data"], sim["subject_ids"], sim["Z_vals"],
                               sim["Z_max"], sim["labels"])
    pooled = _long(vd, sid, Zv, Zm, labels, N_S_max=3)
    psd = pooled._AbstractSustain__sustainData
    rng = np.random.default_rng(42)
    best = None
    for _ in range(15):
        S0 = np.array([pooled._initialise_sequence(psd, rng)[0] for _ in range(3)])
        ml_seq, ml_f, ml_like, *_ = pooled._perform_em(psd, S0, np.ones(3) / 3, rng)
        if best is None or ml_like > best[2]:
            best = (ml_seq, ml_f, ml_like)
    perm, _ = align_subtypes(sim["gt_sequences"], best[0])
    tau = np.mean([seq_tau(sim["gt_sequences"][i], best[0][perm[i]]) for i in range(3)])
    assert tau > 0.85, f"longitudinal recovery tau too low: {tau}"
