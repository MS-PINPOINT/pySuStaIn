"""Federated centres for SuStaIn.

A centre owns its data and a *local SuStaIn model* and returns only **aggregate**
statistics — sums over its own subjects — never per-subject values or raw data.

The base ``FederatedClient`` is **model-agnostic**: it works with any SuStaIn-like
model exposing ``_calculate_likelihood_stage(sustain_data, S)`` (returning a
per-subject ``(M x N+1)`` stage-likelihood) and ``stage_zscore``. Cross-sectional
and longitudinal Z-score models therefore both plug in unchanged — the only
difference is which local model/data a centre is built with.

Convenience subclasses:
  * ``ZscoreFederatedClient``        - cross-sectional Z-score data
  * ``LongitudinalFederatedClient``  - longitudinal (per-subject visits) data
"""
import os
import tempfile
from collections import OrderedDict

import numpy as np

from pySuStaIn.ZscoreSustain import ZscoreSustain, ZScoreSustainData


class FederatedClient:
    """Generic centre wrapping a local model + its data object."""

    def __init__(self, local_model, sustain_data, name="centre", cache_size=512):
        self._model = local_model
        self._data = sustain_data
        self.name = name
        self._N = local_model.stage_zscore.shape[1]      # number of events
        self.M = int(sustain_data.getNumSamples())       # number of subjects
        # Memoise the per-sequence stage-likelihood. It is a pure function of the
        # event ordering (the model params and data are fixed), so caching is
        # numerically exact. Bounded LRU keyed by the integer ordering; keep the
        # default conservative because each cached value is subjects x stages.
        self._cache = OrderedDict()
        self._cache_max = int(cache_size)

    @property
    def num_samples(self):
        return self.M

    def _stage(self, seq):
        """Memoised ``_calculate_likelihood_stage`` for one subtype ordering."""
        key = np.asarray(seq).astype(int).tobytes()
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        val = self._model._calculate_likelihood_stage(self._data, np.asarray(seq))
        self._cache[key] = val
        if len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)
        return val

    # --- per-subject stage likelihoods for all subtypes (local only) ----------
    def _pperm_all(self, S):
        S = np.asarray(S)
        N_S = S.shape[0]
        out = np.zeros((self.M, self._N + 1, N_S))
        for s in range(N_S):
            out[:, :, s] = self._stage(S[s])
        return out

    # --- aggregate statistics returned to the server -------------------------
    def responsibility_sums(self, S, f):
        """(R_c, loglike_c, M_c) with R_c[s] = sum_m r[m,s] (per-subject normalised)."""
        f = np.asarray(f, dtype=float)
        N_S = np.asarray(S).shape[0]
        p = self._pperm_all(S)
        w = p * f.reshape(1, 1, N_S)
        denom = np.sum(w + 1e-250, axis=(1, 2), keepdims=True)
        norm = w / denom
        R = np.sum(norm, axis=(0, 1))
        total_prob_subj = np.sum(w, axis=(1, 2))
        loglike = float(np.sum(np.log(total_prob_subj + 1e-250)))
        return R, loglike, self.M

    def loglike(self, S, f):
        f = np.asarray(f, dtype=float)
        N_S = np.asarray(S).shape[0]
        p = self._pperm_all(S)
        total_prob_subj = np.sum(p * f.reshape(1, 1, N_S), axis=(1, 2))
        return float(np.sum(np.log(total_prob_subj + 1e-250)))

    def score_candidates(self, S_current, f, s, candidate_seqs):
        """Per-candidate partial log-likelihood for moving subtype ``s``."""
        f = np.asarray(f, dtype=float)
        S_current = np.asarray(S_current)
        N_S = S_current.shape[0]
        wsum_others = np.zeros(self.M)
        for sp in range(N_S):
            if sp == s:
                continue
            wsum_others += f[sp] * np.sum(self._stage(S_current[sp]), axis=1)
        scores = np.zeros(len(candidate_seqs))
        for idx, seq in enumerate(candidate_seqs):
            tps = wsum_others + f[s] * np.sum(self._stage(seq), axis=1)
            scores[idx] = np.sum(np.log(tps + 1e-250))
        return scores

    def subtype_and_stage(self, S, f):
        """Local per-subject ML subtype + (baseline) stage."""
        f = np.asarray(f, dtype=float)
        N_S = np.asarray(S).shape[0]
        p = self._pperm_all(S)
        w = p * f.reshape(1, 1, N_S)
        prob_cluster = np.sum(w, axis=1)
        prob_cluster = prob_cluster / np.sum(prob_cluster + 1e-250, axis=1, keepdims=True)
        ml_subtype = np.argmax(prob_cluster, axis=1)
        ml_stage = np.array([int(np.argmax(w[m, :, ml_subtype[m]])) for m in range(self.M)])
        return ml_subtype, ml_stage, prob_cluster

    def subtype_stage_summary(self, S, f):
        """Aggregate local assignment counts, without returning row-level labels."""
        ml_subtype, ml_stage, _ = self.subtype_and_stage(S, f)
        N_S = np.asarray(S).shape[0]
        counts = np.zeros((N_S, self._N + 1), dtype=int)
        np.add.at(counts, (ml_subtype, ml_stage), 1)
        return {
            "num_samples": self.M,
            "subtype_stage_counts": counts,
            "subtype_counts": counts.sum(axis=1),
            "stage_counts": counts.sum(axis=0),
        }


def _mk_output(folder, prefix):
    if folder is None:
        folder = tempfile.mkdtemp(prefix=prefix)
    os.makedirs(folder, exist_ok=True)
    return folder


class ZscoreFederatedClient(FederatedClient):
    """Cross-sectional Z-score centre (one row per subject)."""

    def __init__(self, data, Z_vals, Z_max, biomarker_labels, name="centre",
                 seed=0, output_folder=None, cache_size=512):
        data = np.asarray(data, dtype=float)
        model = ZscoreSustain(
            data, Z_vals, Z_max, biomarker_labels,
            N_startpoints=1, N_S_max=1, N_iterations_MCMC=1,
            output_folder=_mk_output(output_folder, "fed_client_"),
            dataset_name=name, use_parallel_startpoints=False, seed=seed,
        )
        sustain_data = ZScoreSustainData(data, model.stage_zscore.shape[1])
        super().__init__(model, sustain_data, name=name, cache_size=cache_size)


class LongitudinalFederatedClient(FederatedClient):
    """Longitudinal centre (multiple interdependent visits per subject)."""

    def __init__(self, visit_data, subject_ids, Z_vals, Z_max, biomarker_labels,
                 name="centre", seed=0, output_folder=None, cache_size=512):
        # imported here to avoid a hard dependency when only cross-sectional is used
        from pySuStaIn.LongitudinalZscoreSustain import LongitudinalZscoreSustain
        model = LongitudinalZscoreSustain(
            visit_data, subject_ids, Z_vals, Z_max, biomarker_labels,
            N_startpoints=1, N_S_max=1, N_iterations_MCMC=1,
            output_folder=_mk_output(output_folder, "fed_client_long_"),
            dataset_name=name, use_parallel_startpoints=False, seed=seed,
        )
        super().__init__(model, model._AbstractSustain__sustainData, name=name, cache_size=cache_size)
