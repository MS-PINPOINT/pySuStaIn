"""Golden regression: pin the exact current fit outputs so optimisations cannot
silently change the numerics.

These deterministic fits (fixed init + RNG) must keep producing the same
sequences and log-likelihood. They guard the longitudinal monotone-path DP and
the cross-sectional EM independently of the fed==pooled checks (a DP change that
affects pooled and federated equally would pass fed==pooled but fail here).

If you intentionally change the model maths, re-snapshot these values.
"""
import tempfile

import numpy as np

from pySuStaIn.ZscoreSustain import ZscoreSustain
from pySuStaIn.LongitudinalZscoreSustain import LongitudinalZscoreSustain
from pySuStaIn.federated.simulate import simulate_zscore, simulate_longitudinal

GOLDEN_XS_LOGLIKE = -1723.134839679955
GOLDEN_XS_SEQ = [[4, 2, 0, 7, 3, 5, 10, 8, 1, 12, 13, 6, 11, 9, 14],
                 [2, 4, 1, 6, 11, 9, 0, 3, 8, 7, 5, 10, 12, 14, 13]]

GOLDEN_LONG_LOGLIKE = -3182.007273642864
GOLDEN_LONG_SEQ = [[0, 2, 3, 8, 1, 6, 13, 11, 7, 4, 5, 9, 10, 12, 14],
                   [4, 2, 9, 7, 12, 0, 1, 14, 3, 6, 11, 5, 10, 8, 13]]


def test_cross_sectional_golden():
    s = simulate_zscore(n_biomarkers=5, n_samples=200, n_subtypes=2, seed=3)
    xs = ZscoreSustain(s["data"], s["Z_vals"], s["Z_max"], s["labels"], 1, 2, 1,
                       tempfile.mkdtemp(), "x", False, 0)
    sd = xs._AbstractSustain__sustainData
    S0 = np.array([xs._initialise_sequence(sd, np.random.default_rng(3))[0] for _ in range(2)])
    seq, f, ll, *_ = xs._perform_em(sd, S0.copy(), np.ones(2) / 2, np.random.default_rng(7))
    assert abs(float(ll) - GOLDEN_XS_LOGLIKE) < 1e-6
    assert seq.astype(int).tolist() == GOLDEN_XS_SEQ


def test_longitudinal_golden():
    L = simulate_longitudinal(n_biomarkers=5, n_subjects=120, n_subtypes=2, n_visits=3, seed=4)
    lg = LongitudinalZscoreSustain(L["visit_data"], L["subject_ids"], L["Z_vals"], L["Z_max"],
                                   L["labels"], 1, 2, 1, tempfile.mkdtemp(), "l", False, 0)
    lsd = lg._AbstractSustain__sustainData
    S0 = np.array([lg._initialise_sequence(lsd, np.random.default_rng(4))[0] for _ in range(2)])
    seq, f, ll, *_ = lg._perform_em(lsd, S0.copy(), np.ones(2) / 2, np.random.default_rng(9))
    assert abs(float(ll) - GOLDEN_LONG_LOGLIKE) < 1e-6
    assert seq.astype(int).tolist() == GOLDEN_LONG_SEQ
