"""A federated centre for Z-score SuStaIn.

Owns its shard of subjects and a local ``ZscoreSustain`` instance (only used for
its data-independent parameters and the per-subject ``_calculate_likelihood_stage``).
Every method returns *aggregate* statistics only — sums over the centre's own
subjects — never per-subject values or raw data.

The three primitives the server needs:
  * ``responsibility_sums(S, f)`` -> (R_c, loglike_c, M_c) for the f-update + convergence
  * ``loglike(S, f)``            -> scalar log-likelihood over this centre
  * ``score_candidates(...)``    -> per-candidate log-likelihood scalars for the
                                    server-driven greedy sequence search
"""
import os
import tempfile

import numpy as np

from pySuStaIn.ZscoreSustain import ZscoreSustain, ZScoreSustainData


class ZscoreFederatedClient:
    def __init__(self, data, Z_vals, Z_max, biomarker_labels, name="centre", seed=0,
                 output_folder=None):
        data = np.asarray(data, dtype=float)
        self.name = name
        self.M = int(data.shape[0])
        if output_folder is None:
            output_folder = tempfile.mkdtemp(prefix="fed_client_")
        os.makedirs(output_folder, exist_ok=True)
        # A local model: gives us identical (data-independent) params + the
        # per-subject stage likelihood. N_startpoints/MCMC are irrelevant here.
        self._model = ZscoreSustain(
            data, Z_vals, Z_max, biomarker_labels,
            N_startpoints=1, N_S_max=1, N_iterations_MCMC=1,
            output_folder=output_folder, dataset_name=name,
            use_parallel_startpoints=False, seed=seed,
        )
        self._N = self._model.stage_zscore.shape[1]            # number of events
        self._data = ZScoreSustainData(data, self._N)

    @property
    def num_samples(self):
        return self.M

    # --- internal: per-subject stage likelihoods for all subtypes -------------
    def _pperm_all(self, S):
        """p_perm_k for every subtype: shape (M, N+1, N_S). Local only."""
        S = np.asarray(S)
        N_S = S.shape[0]
        out = np.zeros((self.M, self._N + 1, N_S))
        for s in range(N_S):
            out[:, :, s] = self._model._calculate_likelihood_stage(self._data, S[s])
        return out

    # --- aggregate statistics returned to the server -------------------------
    def responsibility_sums(self, S, f):
        """Return (R_c, loglike_c, M_c).

        R_c[s] = sum_m sum_stage  p_perm_k_norm[m, stage, s]   (== sum_m r[m,s]);
        normalisation is per-subject so it is computed entirely locally.
        """
        f = np.asarray(f, dtype=float)
        N_S = np.asarray(S).shape[0]
        p = self._pperm_all(S)                                 # (M, N+1, N_S)
        w = p * f.reshape(1, 1, N_S)
        denom = np.sum(w + 1e-250, axis=(1, 2), keepdims=True)  # per subject
        norm = w / denom
        R = np.sum(norm, axis=(0, 1))                          # (N_S,)
        total_prob_subj = np.sum(w, axis=(1, 2))               # (M,)
        loglike = float(np.sum(np.log(total_prob_subj + 1e-250)))
        return R, loglike, self.M

    def loglike(self, S, f):
        f = np.asarray(f, dtype=float)
        N_S = np.asarray(S).shape[0]
        p = self._pperm_all(S)
        total_prob_subj = np.sum(p * f.reshape(1, 1, N_S), axis=(1, 2))
        return float(np.sum(np.log(total_prob_subj + 1e-250)))

    def score_candidates(self, S_current, f, s, candidate_seqs):
        """Partial log-likelihood for each candidate ordering of subtype ``s``.

        total_prob_subj[m] = sum_{s'} f_{s'} * sum_stage p_perm_k[m, stage, s'].
        The non-``s`` subtypes are fixed by ``S_current`` (computed once); only
        subtype ``s`` is swapped to each candidate. Returns a vector of
        ``sum_m log(total_prob_subj)`` over this centre's subjects.
        """
        f = np.asarray(f, dtype=float)
        S_current = np.asarray(S_current)
        N_S = S_current.shape[0]
        # contribution of the non-moving subtypes (sum over stages, weighted)
        wsum_others = np.zeros(self.M)
        for sp in range(N_S):
            if sp == s:
                continue
            p_sp = self._model._calculate_likelihood_stage(self._data, S_current[sp])
            wsum_others += f[sp] * np.sum(p_sp, axis=1)
        scores = np.zeros(len(candidate_seqs))
        for idx, seq in enumerate(candidate_seqs):
            p_s = self._model._calculate_likelihood_stage(self._data, np.asarray(seq))
            tps = wsum_others + f[s] * np.sum(p_s, axis=1)
            scores[idx] = np.sum(np.log(tps + 1e-250))
        return scores

    def subtype_and_stage(self, S, f):
        """Local per-subject ML subtype + stage (assignment stays at the centre)."""
        f = np.asarray(f, dtype=float)
        N_S = np.asarray(S).shape[0]
        p = self._pperm_all(S)                                 # (M, N+1, N_S)
        w = p * f.reshape(1, 1, N_S)
        prob_cluster = np.sum(w, axis=1)                       # (M, N_S)
        prob_cluster = prob_cluster / np.sum(prob_cluster + 1e-250, axis=1, keepdims=True)
        ml_subtype = np.argmax(prob_cluster, axis=1)
        # stage within the ML subtype
        ml_stage = np.zeros(self.M, dtype=int)
        for m in range(self.M):
            s = ml_subtype[m]
            ml_stage[m] = int(np.argmax(w[m, :, s]))
        return ml_subtype, ml_stage, prob_cluster
