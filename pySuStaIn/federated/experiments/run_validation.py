"""Validate federated Z-score SuStaIn against pooled SuStaIn on simulated data.

Two checks:
  1. EQUIVALENCE  - from an identical initial sequence + RNG, one federated EM
     optimisation must reproduce pooled SuStaIn's EM (sequences, f, log-like).
  2. RECOVERY     - multi-start federated fit recovers the ground-truth subtype
     sequences, and agrees with an equivalent pooled multi-start fit.

Federation is simulated in-process across N centres (no network); each centre
only ever returns aggregate statistics.
"""
import argparse
import tempfile

import numpy as np
from scipy.stats import kendalltau

from pySuStaIn.ZscoreSustain import ZscoreSustain
from pySuStaIn.federated.client import ZscoreFederatedClient
from pySuStaIn.federated.server import FederatedZscoreSustain
from pySuStaIn.federated.simulate import simulate_zscore, split_into_centres


def make_pooled(data, Z_vals, Z_max, labels, seed=0):
    return ZscoreSustain(
        data, Z_vals, Z_max, labels,
        N_startpoints=1, N_S_max=2, N_iterations_MCMC=1,
        output_folder=tempfile.mkdtemp(prefix="pooled_"), dataset_name="pooled",
        use_parallel_startpoints=False, seed=seed,
    )


def pooled_perform_em(pooled, S_init, f_init, rng):
    sd = pooled._AbstractSustain__sustainData
    ml_seq, ml_f, ml_like, *_ = pooled._perform_em(sd, S_init, f_init, rng)
    return ml_seq, ml_f, ml_like


def pooled_fit(pooled, N_S, n_startpoints, seed):
    rng = np.random.default_rng(seed)
    sd = pooled._AbstractSustain__sustainData
    best = None
    for _ in range(n_startpoints):
        S0 = np.array([pooled._initialise_sequence(sd, rng)[0] for _ in range(N_S)])
        f0 = np.ones(N_S) / N_S
        ml_seq, ml_f, ml_like, *_ = pooled._perform_em(sd, S0, f0, rng)
        if best is None or ml_like > best[2]:
            best = (ml_seq, ml_f, ml_like)
    return best


def seq_tau(a, b):
    """Kendall tau between two event orderings (via event positions)."""
    a, b = np.asarray(a), np.asarray(b)
    N = len(a)
    pa = np.zeros(N); pa[a.astype(int)] = np.arange(N)
    pb = np.zeros(N); pb[b.astype(int)] = np.arange(N)
    return kendalltau(pa, pb).correlation


def align_subtypes(S_ref, S_test):
    """Best permutation of S_test subtypes to S_ref by mean Kendall tau (small N_S)."""
    from itertools import permutations
    N_S = S_ref.shape[0]
    best_perm, best_score = None, -np.inf
    for perm in permutations(range(N_S)):
        score = np.mean([seq_tau(S_ref[i], S_test[perm[i]]) for i in range(N_S)])
        if score > best_score:
            best_score, best_perm = score, perm
    return list(best_perm), best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-biomarkers", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-subtypes", type=int, default=2)
    ap.add_argument("--n-centres", type=int, default=10)
    ap.add_argument("--startpoints", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--heterogeneous", action="store_true",
                    help="give centres heterogeneous subtype prevalence")
    args = ap.parse_args()

    print(f"[sim] {args.n_samples} subjects, {args.n_biomarkers} biomarkers, "
          f"{args.n_subtypes} subtypes, {args.n_centres} centres")
    sim = simulate_zscore(args.n_biomarkers, args.n_samples, args.n_subtypes,
                          subtype_fractions=[0.6, 0.4] if args.n_subtypes == 2 else None,
                          seed=args.seed)
    data, Z_vals, Z_max, labels = sim["data"], sim["Z_vals"], sim["Z_max"], sim["labels"]
    gt = sim["gt_sequences"]

    shard_idx = split_into_centres(
        data.shape[0], args.n_centres, seed=args.seed,
        group=sim["gt_subtypes"] if args.heterogeneous else None,
        concentration=0.3 if args.heterogeneous else None,
    )
    print(f"[sim] centre sizes: {[len(s) for s in shard_idx]}")

    pooled = make_pooled(data, Z_vals, Z_max, labels, seed=args.seed)
    clients = [ZscoreFederatedClient(data[idx], Z_vals, Z_max, labels, name=f"centre{i}")
               for i, idx in enumerate(shard_idx)]
    fed = FederatedZscoreSustain(clients, Z_vals, Z_max, labels,
                                 N_startpoints=args.startpoints, N_S_max=args.n_subtypes)

    # ---------------- 1. EQUIVALENCE ----------------
    print("\n=== 1. EQUIVALENCE: federated EM vs pooled EM (same init + RNG) ===")
    init_rng = np.random.default_rng(args.seed)
    sd = pooled._AbstractSustain__sustainData
    S0 = np.array([pooled._initialise_sequence(sd, init_rng)[0] for _ in range(args.n_subtypes)])
    f0 = np.ones(args.n_subtypes) / args.n_subtypes

    pooled_seq, pooled_f, pooled_like = pooled_perform_em(
        pooled, S0.copy(), f0.copy(), np.random.default_rng(args.seed))
    fed_seq, fed_f, fed_like = fed.fit_em(
        S0.copy(), f0.copy(), np.random.default_rng(args.seed))

    seq_match = np.array_equal(np.asarray(pooled_seq).astype(int), np.asarray(fed_seq).astype(int))
    max_seq_diff = int(np.max(np.abs(np.asarray(pooled_seq).astype(int) - np.asarray(fed_seq).astype(int))))
    print(f"  pooled loglike = {pooled_like:.6f}")
    print(f"  fed    loglike = {fed_like:.6f}")
    print(f"  |loglike diff| = {abs(pooled_like - fed_like):.3e}")
    print(f"  |f diff|       = {np.max(np.abs(np.asarray(pooled_f)-np.asarray(fed_f))):.3e}")
    print(f"  sequences identical = {seq_match} (max element diff {max_seq_diff})")
    equivalence_ok = abs(pooled_like - fed_like) < 1e-6 and seq_match

    # ---------------- 2. RECOVERY ----------------
    print(f"\n=== 2. RECOVERY: multi-start fit (N_S={args.n_subtypes}, "
          f"{args.startpoints} starts) vs ground truth ===")
    fed_best = fed.fit(args.n_subtypes, n_startpoints=args.startpoints, seed=args.seed)
    pooled_best = pooled_fit(pooled, args.n_subtypes, args.startpoints, seed=args.seed)
    fed_S, fed_f2, fed_like2 = fed_best
    pooled_S, pooled_f2, pooled_like2 = pooled_best

    perm_fed, _ = align_subtypes(gt, fed_S)
    perm_pool, _ = align_subtypes(gt, pooled_S)
    tau_fed = [seq_tau(gt[i], fed_S[perm_fed[i]]) for i in range(args.n_subtypes)]
    tau_pool = [seq_tau(gt[i], pooled_S[perm_pool[i]]) for i in range(args.n_subtypes)]
    # federated vs pooled (align pooled to fed)
    perm_fp, _ = align_subtypes(fed_S, pooled_S)
    tau_fp = [seq_tau(fed_S[i], pooled_S[perm_fp[i]]) for i in range(args.n_subtypes)]

    print(f"  fed    loglike = {fed_like2:.4f}   fractions = {np.round(np.sort(fed_f2)[::-1],3)}")
    print(f"  pooled loglike = {pooled_like2:.4f}   fractions = {np.round(np.sort(pooled_f2)[::-1],3)}")
    print(f"  Kendall tau  federated vs ground truth : {np.round(tau_fed,3)}  (mean {np.mean(tau_fed):.3f})")
    print(f"  Kendall tau  pooled    vs ground truth : {np.round(tau_pool,3)}  (mean {np.mean(tau_pool):.3f})")
    print(f"  Kendall tau  federated vs pooled       : {np.round(tau_fp,3)}  (mean {np.mean(tau_fp):.3f})")

    print("\n=== SUMMARY ===")
    print(f"  equivalence (fed EM == pooled EM): {'PASS' if equivalence_ok else 'FAIL'}")
    print(f"  recovery (fed mean tau vs truth) : {np.mean(tau_fed):.3f}")
    print(f"  fed vs pooled agreement (mean tau): {np.mean(tau_fp):.3f}")
    return 0 if equivalence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
