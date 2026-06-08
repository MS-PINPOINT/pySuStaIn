"""Longitudinal Z-score SuStaIn.

Cross-sectional SuStaIn treats every row as an independent subject. Here a
*subject* has multiple visits that are **interdependent**: they share a single
subtype, and their SuStaIn stage is **monotonically non-decreasing** over time
(disease does not reverse). This is the natural way to feed longitudinal data in.

Key idea — it changes *only* the per-subject likelihood:

    p(subject m | subtype s) = sum over monotone stage paths k_1 <= ... <= k_V
                                  of  prod_v  e_v(k_v)

where e_v(k) is the per-visit Z-score emission (exactly the cross-sectional
``ZscoreSustain._calculate_likelihood_stage`` applied to that visit). We compute
this with an O(V * N) dynamic program and return, per subject, the *joint with
the baseline stage*  J(k) = e_1(k) * g_2(k), whose sum over k equals the
monotone-path marginal. Because SuStaIn always reduces ``_calculate_likelihood_stage``
by summing over the stage axis, this slots into the unchanged EM / greedy
sequence search / MCMC: a SuStaIn model whose data happen to be interdependent
across visits. With one visit per subject it is *identical* to cross-sectional
SuStaIn.

Data layout: ``visit_data`` is (n_visits x n_biomarkers) of positive Z-scores,
``subject_ids`` (n_visits,) groups visits by subject, and rows for a subject are
assumed to be in chronological order.
"""
from collections import defaultdict

import numpy as np

from pySuStaIn.AbstractSustain import AbstractSustainData
from pySuStaIn.ZscoreSustain import ZscoreSustain, ZScoreSustainData


class LongitudinalZScoreSustainData(AbstractSustainData):
    def __init__(self, visit_data, subject_ids, numStages):
        self.visit_data = np.asarray(visit_data, dtype=float)
        self.subject_ids = np.asarray(subject_ids)
        if self.visit_data.ndim != 2:
            raise ValueError("visit_data must be a 2D array of shape (visits, biomarkers)")
        if self.subject_ids.shape[0] != self.visit_data.shape[0]:
            raise ValueError("subject_ids must have one entry per visit_data row")
        self.__numStages = numStages
        # group visit rows by subject, preserving first-appearance order and the
        # within-subject row order (assumed chronological)
        order = {}
        groups = []
        for r, sid in enumerate(self.subject_ids):
            if sid not in order:
                order[sid] = len(groups)
                groups.append([])
            groups[order[sid]].append(r)
        self.subject_groups = [np.array(g, dtype=int) for g in groups]

    def getNumSamples(self):
        return len(self.subject_groups)

    def getNumBiomarkers(self):
        return self.visit_data.shape[1]

    def getNumStages(self):
        return self.__numStages

    def reindex(self, index):
        """Subset *subjects* (index into subjects, not visit rows)."""
        index = np.asarray(index)
        keep_groups = [self.subject_groups[i] for i in index]
        rows = np.concatenate(keep_groups) if len(keep_groups) else np.array([], dtype=int)
        new_visit_data = self.visit_data[rows]
        new_subject_ids = self.subject_ids[rows]
        return LongitudinalZScoreSustainData(new_visit_data, new_subject_ids, self.__numStages)


class LongitudinalZscoreSustain(ZscoreSustain):
    def __init__(self, visit_data, subject_ids, Z_vals, Z_max, biomarker_labels,
                 N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
                 dataset_name, use_parallel_startpoints, seed=None):
        visit_data = np.asarray(visit_data, dtype=float)
        # Reuse the parent initialiser to build the (data-independent) Z-score
        # parameters; it also builds a cross-sectional __sustainData over the
        # visit rows, which we then replace with the longitudinal one.
        super().__init__(
            visit_data, Z_vals, Z_max, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed,
        )
        numStages = self.stage_zscore.shape[1]
        self._AbstractSustain__sustainData = LongitudinalZScoreSustainData(
            visit_data, subject_ids, numStages
        )

    def _calculate_likelihood_stage(self, sustainData, S):
        """Per-subject joint with baseline stage, via the monotone-path DP.

        Returns (n_subjects x N+1). Summed over the stage axis this is the
        monotone-path marginal p(subject | subtype with ordering S).
        """
        # Per-visit emissions: reuse the parent cross-sectional computation on
        # the visit rows (identical Z-score Gaussian model).
        numStages = self.stage_zscore.shape[1]
        visit_obj = ZScoreSustainData(sustainData.visit_data, numStages)
        E = ZscoreSustain._calculate_likelihood_stage(self, visit_obj, S)  # (n_visits, N+1)

        groups = sustainData.subject_groups
        M = len(groups)
        Np1 = E.shape[1]
        J = np.zeros((M, Np1))
        # Run the monotone-path DP vectorised over subjects, grouped by their
        # number of visits V (same operations as the per-subject recursion,
        # batched). g_v(k) = sum_{k'>=k} e_v(k') g_{v+1}(k') is a reverse cumsum.
        by_v = defaultdict(list)
        for m, rows in enumerate(groups):
            by_v[len(rows)].append(m)
        for V, subj_idx in by_v.items():
            subj_idx = np.asarray(subj_idx)
            A = np.stack([E[groups[m]] for m in subj_idx], axis=0)  # (n_g, V, N+1)
            g = np.ones((len(subj_idx), Np1))                       # g_{V+1}
            for v in range(V - 1, 0, -1):                           # visits V..2
                cont = A[:, v, :] * g
                g = np.cumsum(cont[:, ::-1], axis=1)[:, ::-1]
            J[subj_idx] = A[:, 0, :] * g                            # joint with baseline stage
        return J
