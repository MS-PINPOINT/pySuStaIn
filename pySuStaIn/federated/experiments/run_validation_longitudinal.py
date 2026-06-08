"""Validate longitudinal SuStaIn (pooled and federated) on simulated data.

Subjects have multiple interdependent visits (one subtype, monotone stage).
Checks:
  1. EQUIVALENCE - federated-longitudinal EM == pooled-longitudinal EM (same
     init + RNG): identical sequences, fractions, log-likelihood.
  2. RECOVERY    - multi-start longitudinal fit recovers the ground-truth subtype
     sequences (3 subtypes, 10 biomarkers) and agrees with pooled.
"""
import argparse
import tempfile

import numpy as np

from pySuStaIn.LongitudinalZscoreSustain import LongitudinalZscoreSustain
from pySuStaIn.federated.client import LongitudinalFederatedClient
from pySuStaIn.federated.server import FederatedZscoreSustain
from pySuStaIn.federated.simulate import simulate_longitudinal, split_subjects_into_centres
from pySuStaIn.federated.experiments.run_validation import seq_tau, align_subtypes


def pooled_long(visit_data, subject_ids, Zv, Zm, labels, N_S_max=3, seed=0):
    return LongitudinalZscoreSustain(
        visit_data, subject_ids, Zv, Zm, labels,
        N_startpoints=1, N_S_max=N_S_max, N_iterations_MCMC=1,
        output_folder=tempfile.mkdtemp(prefix="pooled_long_"),
        dataset_name="pooled_long", use_parallel_startpoints=False, seed=seed,
    )


def multi_start(model_perform_em, init_fn, N_S, n_startpoints, seed):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_startpoints):
        S0 = np.array([init_fn(rng)[0] for _ in range(N_S)])
        f0 = np.ones(N_S) / N_S
        ml_seq, ml_f, ml_like = model_perform_em(S0, f0, rng)
        if best is None or ml_like > best[2]:
            best = (ml_seq, ml_f, ml_like)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-biomarkers", type=int, default=10)
    ap.add_argument("--n-subjects", type=int, default=600)
    ap.add_argument("--n-subtypes", type=int, default=3)
    ap.add_argument("--n-visits", type=int, default=3)
    ap.add_argument("--n-centres", type=int, default=10)
    ap.add_argument("--startpoints", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[sim] {args.n_subjects} subjects x {args.n_visits} visits, "
          f"{args.n_biomarkers} biomarkers, {args.n_subtypes} subtypes, {args.n_centres} centres")
    sim = simulate_longitudinal(args.n_biomarkers, args.n_subjects, args.n_subtypes,
                                args.n_visits, seed=args.seed)
    vd, sid, Zv, Zm, labels = (sim["visit_data"], sim["subject_ids"], sim["Z_vals"],
                               sim["Z_max"], sim["labels"])
    gt = sim["gt_sequences"]

    pooled = pooled_long(vd, sid, Zv, Zm, labels, N_S_max=args.n_subtypes, seed=args.seed)
    psd = pooled._AbstractSustain__sustainData

    shard_rows = split_subjects_into_centres(sid, args.n_centres, seed=args.seed)
    clients = [LongitudinalFederatedClient(vd[rows], sid[rows], Zv, Zm, labels, name=f"c{i}")
               for i, rows in enumerate(shard_rows)]
    fed = FederatedZscoreSustain(clients, Zv, Zm, labels, N_S_max=args.n_subtypes)
    print(f"[sim] centre subject counts: "
          f"{[len(np.unique(sid[r])) for r in shard_rows]}")

    # ---- 1. EQUIVALENCE ----
    print("\n=== 1. EQUIVALENCE: federated-long EM vs pooled-long EM (same init+RNG) ===")
    init_rng = np.random.default_rng(args.seed)
    S0 = np.array([pooled._initialise_sequence(psd, init_rng)[0] for _ in range(args.n_subtypes)])
    f0 = np.ones(args.n_subtypes) / args.n_subtypes
    ps, pf, pl, *_ = pooled._perform_em(psd, S0.copy(), f0.copy(), np.random.default_rng(args.seed))
    fs, ff, fl = fed.fit_em(S0.copy(), f0.copy(), np.random.default_rng(args.seed))
    seq_match = np.array_equal(np.asarray(ps).astype(int), np.asarray(fs).astype(int))
    print(f"  pooled loglike = {pl:.6f}")
    print(f"  fed    loglike = {fl:.6f}")
    print(f"  |loglike diff| = {abs(pl-fl):.3e}   |f diff| = {np.max(np.abs(pf-ff)):.3e}")
    print(f"  sequences identical = {seq_match}")
    equivalence_ok = abs(pl - fl) < 1e-6 and seq_match

    # ---- 2. RECOVERY ----
    print(f"\n=== 2. RECOVERY: {args.n_subtypes}-subtype fit ({args.startpoints} starts) vs truth ===")
    fed_S, fed_f, fed_like = multi_start(
        lambda S0, f0, rng: fed.fit_em(S0, f0, rng),
        lambda rng: pooled._initialise_sequence(psd, rng), args.n_subtypes, args.startpoints, args.seed)
    pooled_S, pooled_f, pooled_like = multi_start(
        lambda S0, f0, rng: pooled._perform_em(psd, S0, f0, rng)[:3],
        lambda rng: pooled._initialise_sequence(psd, rng), args.n_subtypes, args.startpoints, args.seed)

    perm_fed, _ = align_subtypes(gt, fed_S)
    perm_pool, _ = align_subtypes(gt, pooled_S)
    perm_fp, _ = align_subtypes(fed_S, pooled_S)
    tau_fed = [seq_tau(gt[i], fed_S[perm_fed[i]]) for i in range(args.n_subtypes)]
    tau_pool = [seq_tau(gt[i], pooled_S[perm_pool[i]]) for i in range(args.n_subtypes)]
    tau_fp = [seq_tau(fed_S[i], pooled_S[perm_fp[i]]) for i in range(args.n_subtypes)]
    print(f"  fed    loglike = {fed_like:.3f}   fractions = {np.round(np.sort(fed_f)[::-1],3)}")
    print(f"  pooled loglike = {pooled_like:.3f}")
    print(f"  Kendall tau  fed vs truth   : {np.round(tau_fed,3)}  (mean {np.mean(tau_fed):.3f})")
    print(f"  Kendall tau  pooled vs truth: {np.round(tau_pool,3)}  (mean {np.mean(tau_pool):.3f})")
    print(f"  Kendall tau  fed vs pooled  : {np.round(tau_fp,3)}  (mean {np.mean(tau_fp):.3f})")

    print("\n=== SUMMARY ===")
    print(f"  equivalence (fed-long == pooled-long): {'PASS' if equivalence_ok else 'FAIL'}")
    print(f"  recovery (fed mean tau vs truth)     : {np.mean(tau_fed):.3f}")
    print(f"  fed vs pooled agreement (mean tau)   : {np.mean(tau_fp):.3f}")
    return 0 if equivalence_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
