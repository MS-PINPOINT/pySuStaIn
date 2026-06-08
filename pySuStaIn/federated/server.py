"""Federated Z-score SuStaIn server / orchestrator.

Subclasses ``ZscoreSustain`` and overrides the two data-touching primitives
(``_calculate_likelihood`` and ``_optimise_parameters``) so they aggregate over
``ZscoreFederatedClient`` centres instead of reading a local data matrix. The
candidate-generation logic of the greedy sequence search is reproduced verbatim
from pooled ``ZscoreSustain._optimise_parameters`` (it is data-independent); only
the likelihood evaluation is replaced by a sum of per-centre scalars.

With the same RNG and initial sequence, the federated EM makes identical
f-updates and sequence moves as pooled SuStaIn -> the fits coincide.
"""
import os
import tempfile

import numpy as np

from pySuStaIn.ZscoreSustain import ZscoreSustain


class FederatedZscoreSustain(ZscoreSustain):
    def __init__(self, clients, Z_vals, Z_max, biomarker_labels,
                 N_startpoints=25, N_S_max=1, N_iterations_MCMC=int(1e4),
                 output_folder=None, dataset_name="federated",
                 use_parallel_startpoints=False, seed=None):
        if output_folder is None:
            output_folder = tempfile.mkdtemp(prefix="fed_server_")
        os.makedirs(output_folder, exist_ok=True)
        # Dummy data only to build the (data-independent) model parameters; the
        # server never reads it — all data lives at the clients.
        n_biomarkers = len(biomarker_labels)
        dummy = np.ones((2, n_biomarkers))
        super().__init__(
            dummy, Z_vals, Z_max, biomarker_labels,
            N_startpoints, N_S_max, N_iterations_MCMC,
            output_folder, dataset_name, use_parallel_startpoints, seed,
        )
        self.clients = list(clients)
        self.N_total = int(sum(c.num_samples for c in self.clients))
        self._N = self.stage_zscore.shape[1]

    # ------------------------------------------------------------------ #
    # Aggregation helpers
    # ------------------------------------------------------------------ #
    def _agg_responsibilities(self, S, f):
        R = None
        loglike = 0.0
        for c in self.clients:
            R_c, ll_c, _ = c.responsibility_sums(S, f)
            R = R_c if R is None else R + R_c
            loglike += ll_c
        return R, loglike

    def _agg_loglike(self, S, f):
        return float(sum(c.loglike(S, f) for c in self.clients))

    def _agg_candidate_scores(self, S_current, f, s, candidate_seqs):
        scores = np.zeros(len(candidate_seqs))
        for c in self.clients:
            scores += c.score_candidates(S_current, f, s, candidate_seqs)
        return scores

    # ------------------------------------------------------------------ #
    # Overridden SuStaIn primitives (used by the inherited _perform_em)
    # ------------------------------------------------------------------ #
    def _calculate_likelihood(self, sustainData, S, f):
        loglike = self._agg_loglike(S, np.asarray(f).reshape(-1))
        # _perform_em only consumes the log-likelihood; per-subject arrays stay
        # at the centres.
        return loglike, None, None, None, None

    def _candidate_sequences(self, S_opt, s, selected_event):
        """Reproduces pooled ZscoreSustain candidate generation (data-independent)."""
        N = self._N
        current_sequence = S_opt[s]
        current_location = np.array([0] * len(current_sequence))
        current_location[current_sequence.astype(int)] = np.arange(len(current_sequence))

        move_event_from = current_location[selected_event]
        this_stage_zscore = self.stage_zscore[0, selected_event]
        selected_biomarker = self.stage_biomarker_index[0, selected_event]
        possible_zscores_biomarker = self.stage_zscore[self.stage_biomarker_index == selected_biomarker]

        min_filter = possible_zscores_biomarker < this_stage_zscore
        max_filter = possible_zscores_biomarker > this_stage_zscore
        events = np.array(range(N))
        if np.any(min_filter):
            min_zscore_bound = max(possible_zscores_biomarker[min_filter])
            min_zscore_bound_event = events[((self.stage_zscore[0] == min_zscore_bound).astype(int) +
                                             (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
            move_event_to_lower_bound = current_location[min_zscore_bound_event] + 1
        else:
            move_event_to_lower_bound = 0
        if np.any(max_filter):
            max_zscore_bound = min(possible_zscores_biomarker[max_filter])
            max_zscore_bound_event = events[((self.stage_zscore[0] == max_zscore_bound).astype(int) +
                                             (self.stage_biomarker_index[0] == selected_biomarker).astype(int)) == 2]
            move_event_to_upper_bound = current_location[max_zscore_bound_event]
        else:
            move_event_to_upper_bound = N
        if move_event_to_lower_bound == move_event_to_upper_bound:
            possible_positions = np.array([0])
        else:
            possible_positions = np.arange(move_event_to_lower_bound, move_event_to_upper_bound)

        possible_sequences = np.zeros((len(possible_positions), N))
        for index in range(len(possible_positions)):
            current_sequence = S_opt[s]
            move_event_to = possible_positions[index]
            current_sequence = np.delete(current_sequence, move_event_from, 0)
            new_sequence = np.concatenate([current_sequence[np.arange(move_event_to)],
                                           [selected_event],
                                           current_sequence[np.arange(move_event_to, N - 1)]])
            possible_sequences[index, :] = new_sequence
        return possible_sequences

    def _optimise_parameters(self, sustainData, S_init, f_init, rng):
        N = self._N
        N_S = S_init.shape[0]
        S_opt = S_init.copy()

        # initial f-update (aggregate responsibilities), matches pooled
        R, _ = self._agg_responsibilities(S_opt, np.asarray(f_init).reshape(-1))
        f_opt = (R / R.sum()).reshape(-1)

        order_seq = rng.permutation(N_S)
        for s in order_seq:
            order_bio = rng.permutation(N)
            for i in order_bio:
                possible_sequences = self._candidate_sequences(S_opt, s, i)
                scores = self._agg_candidate_scores(S_opt, f_opt, s, possible_sequences)
                best = possible_sequences[scores == scores.max(), :][0, :]
                S_opt[s] = best

        # final f-update + likelihood (aggregate)
        R, _ = self._agg_responsibilities(S_opt, f_opt)
        f_opt = (R / R.sum()).reshape(-1)
        likelihood_opt = self._agg_loglike(S_opt, f_opt)
        return S_opt, f_opt, likelihood_opt

    # ------------------------------------------------------------------ #
    # Convenience fitting entry points
    # ------------------------------------------------------------------ #
    def fit_em(self, S_init, f_init, rng):
        """One EM optimisation to convergence from a given init (federated)."""
        sd = self._AbstractSustain__sustainData
        ml_seq, ml_f, ml_like, *_ = self._perform_em(sd, S_init, f_init, rng)
        return ml_seq, ml_f, ml_like

    def fit(self, N_S, n_startpoints=25, seed=0):
        """Federated ML fit for a fixed number of subtypes via multi-start EM.

        Start-points use the data-independent ``_initialise_sequence`` (identical
        on server and centres). Returns (best_sequence, best_f, best_loglike).
        """
        rng = np.random.default_rng(seed)
        sd = self._AbstractSustain__sustainData
        best = None
        for _ in range(n_startpoints):
            S0 = np.array([self._initialise_sequence(sd, rng)[0] for _ in range(N_S)])
            f0 = np.ones(N_S) / N_S
            ml_seq, ml_f, ml_like = self.fit_em(S0, f0, rng)
            if best is None or ml_like > best[2]:
                best = (ml_seq, ml_f, ml_like)
        return best

    def run_sustain_algorithm(self, *args, **kwargs):
        raise NotImplementedError(
            "FederatedZscoreSustain currently supports ML fitting via fit() and "
            "fit_em() only. The inherited run_sustain_algorithm() also runs "
            "MCMC uncertainty and per-subject central staging, which are not "
            "implemented for the aggregate-only federated path."
        )

    def subtype_and_stage(self, S, f, *, return_individual=False):
        """Per-centre subtype/stage summaries by default.

        Individual assignments are row-level outputs. Keep them at the centre in
        a real federation; ``return_individual=True`` is only for local
        in-process simulations or debugging where row-level return is allowed.
        """
        f = np.asarray(f).reshape(-1)
        if return_individual:
            return {c.name: c.subtype_and_stage(S, f) for c in self.clients}
        return {c.name: c.subtype_stage_summary(S, f) for c in self.clients}
