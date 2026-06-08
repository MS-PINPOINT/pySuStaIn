"""Federated Z-score SuStaIn must reproduce pooled SuStaIn exactly.

Fast, deterministic equivalence check: from an identical initial sequence and
RNG, one federated EM optimisation and one pooled EM optimisation must give the
same sequences, fractions and log-likelihood.
"""
import tempfile

import numpy as np

from pySuStaIn.ZscoreSustain import ZscoreSustain
from pySuStaIn.federated.client import ZscoreFederatedClient
from pySuStaIn.federated.server import FederatedZscoreSustain
from pySuStaIn.federated.simulate import simulate_zscore, split_into_centres


def _pooled(data, Zv, Zm, labels):
    return ZscoreSustain(data, Zv, Zm, labels, 1, 2, 1,
                         tempfile.mkdtemp(prefix="pooled_test_"), "p", False, 0)


def test_federated_equals_pooled_em():
    sim = simulate_zscore(n_biomarkers=5, n_samples=200, n_subtypes=2, seed=3)
    data, Zv, Zm, labels = sim["data"], sim["Z_vals"], sim["Z_max"], sim["labels"]
    shards = split_into_centres(data.shape[0], n_centres=5, seed=3)

    pooled = _pooled(data, Zv, Zm, labels)
    clients = [ZscoreFederatedClient(data[ix], Zv, Zm, labels, name=f"c{i}")
               for i, ix in enumerate(shards)]
    fed = FederatedZscoreSustain(clients, Zv, Zm, labels, N_S_max=2)

    sd = pooled._AbstractSustain__sustainData
    rng0 = np.random.default_rng(3)
    S0 = np.array([pooled._initialise_sequence(sd, rng0)[0] for _ in range(2)])
    f0 = np.ones(2) / 2

    ps, pf, pl, *_ = pooled._perform_em(sd, S0.copy(), f0.copy(), np.random.default_rng(7))
    fs, ff, fl = fed.fit_em(S0.copy(), f0.copy(), np.random.default_rng(7))

    assert abs(pl - fl) < 1e-6, f"loglike differ: {pl} vs {fl}"
    assert np.array_equal(ps.astype(int), fs.astype(int)), "sequences differ"
    assert np.allclose(pf, ff, atol=1e-10), f"fractions differ: {pf} vs {ff}"


def test_single_subtype_equivalence():
    sim = simulate_zscore(n_biomarkers=5, n_samples=150, n_subtypes=1, seed=11)
    data, Zv, Zm, labels = sim["data"], sim["Z_vals"], sim["Z_max"], sim["labels"]
    shards = split_into_centres(data.shape[0], n_centres=4, seed=11)
    pooled = _pooled(data, Zv, Zm, labels)
    clients = [ZscoreFederatedClient(data[ix], Zv, Zm, labels, name=f"c{i}")
               for i, ix in enumerate(shards)]
    fed = FederatedZscoreSustain(clients, Zv, Zm, labels, N_S_max=1)
    sd = pooled._AbstractSustain__sustainData
    S0 = np.array([pooled._initialise_sequence(sd, np.random.default_rng(11))[0]])
    f0 = np.ones(1)
    ps, pf, pl, *_ = pooled._perform_em(sd, S0.copy(), f0.copy(), np.random.default_rng(5))
    fs, ff, fl = fed.fit_em(S0.copy(), f0.copy(), np.random.default_rng(5))
    assert abs(pl - fl) < 1e-6
    assert np.array_equal(ps.astype(int), fs.astype(int))
