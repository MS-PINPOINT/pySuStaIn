"""Ground-truth simulation + centre splitting for federated SuStaIn experiments.

Wraps ``ZscoreSustain.generate_random_model`` / ``generate_data`` to produce a
data set with known subtype sequences, per-subject subtypes and stages, then
splits subjects across centres (optionally with heterogeneous subtype
prevalence per centre).
"""
import numpy as np

from pySuStaIn.ZscoreSustain import ZscoreSustain


def simulate_zscore(n_biomarkers=10, n_samples=1000, n_subtypes=2,
                    subtype_fractions=None, frac_controls=0.25,
                    n_zscores=3, zmax=5, seed=42):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # generate_random_model/generate_data use np.random

    Z_vals = np.tile(np.arange(1, n_zscores + 1), (n_biomarkers, 1)).astype(float)
    Z_max = np.full((n_biomarkers,), float(zmax))

    gt_sequences = ZscoreSustain.generate_random_model(Z_vals, n_subtypes)
    N_stages = int(np.sum(Z_vals > 0) + 1)

    if subtype_fractions is None:
        subtype_fractions = np.ones(n_subtypes) / n_subtypes
    subtype_fractions = np.asarray(subtype_fractions, float)
    subtype_fractions = subtype_fractions / subtype_fractions.sum()
    gt_subtypes = rng.choice(n_subtypes, size=n_samples, p=subtype_fractions)

    n_controls = int(round(n_samples * frac_controls))
    gt_stages = np.zeros((n_samples, 1), dtype=int)
    gt_stages[n_controls:, 0] = rng.integers(1, N_stages + 1, size=n_samples - n_controls)

    data, data_denoised, stage_value = ZscoreSustain.generate_data(
        gt_subtypes, gt_stages, gt_sequences, Z_vals, Z_max
    )

    return {
        "data": np.asarray(data, dtype=float),
        "gt_subtypes": np.asarray(gt_subtypes).reshape(-1),
        "gt_stages": np.asarray(gt_stages).reshape(-1),
        "gt_sequences": np.asarray(gt_sequences),
        "Z_vals": Z_vals,
        "Z_max": Z_max,
        "labels": [f"BM{i}" for i in range(n_biomarkers)],
        "N_stages": N_stages,
    }


def simulate_longitudinal(n_biomarkers=10, n_subjects=600, n_subtypes=3, n_visits=3,
                          subtype_fractions=None, frac_controls=0.2,
                          n_zscores=3, zmax=5, mean_step=4.0, seed=42):
    """Longitudinal ground-truth: each subject has ``n_visits`` interdependent
    visits sharing one subtype and a monotonically non-decreasing stage.

    Returns visit-level data + subject ids + per-subject ground truth.
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    Z_vals = np.tile(np.arange(1, n_zscores + 1), (n_biomarkers, 1)).astype(float)
    Z_max = np.full((n_biomarkers,), float(zmax))
    gt_sequences = ZscoreSustain.generate_random_model(Z_vals, n_subtypes)
    N_stages = int(np.sum(Z_vals > 0) + 1)

    if subtype_fractions is None:
        subtype_fractions = np.ones(n_subtypes) / n_subtypes
    subtype_fractions = np.asarray(subtype_fractions, float)
    subtype_fractions = subtype_fractions / subtype_fractions.sum()
    gt_subtypes = rng.choice(n_subtypes, size=n_subjects, p=subtype_fractions)

    n_controls = int(round(n_subjects * frac_controls))
    is_control = np.zeros(n_subjects, dtype=bool)
    is_control[:n_controls] = True
    rng.shuffle(is_control)

    # per-subject monotone stage trajectory across visits
    subj_stages = np.zeros((n_subjects, n_visits), dtype=int)
    baseline = np.zeros(n_subjects, dtype=int)
    for i in range(n_subjects):
        if is_control[i]:
            continue
        k = int(rng.integers(1, max(2, N_stages - n_visits * 2)))
        baseline[i] = k
        traj = [k]
        for _ in range(n_visits - 1):
            k = min(N_stages, k + int(rng.poisson(mean_step)))
            traj.append(k)
        subj_stages[i] = traj

    # expand to visit level
    subtypes_visit = np.repeat(gt_subtypes, n_visits)
    subject_ids = np.repeat(np.arange(n_subjects), n_visits)
    stages_visit = subj_stages.reshape(-1, 1)

    visit_data, _, _ = ZscoreSustain.generate_data(
        subtypes_visit, stages_visit, gt_sequences, Z_vals, Z_max
    )

    return {
        "visit_data": np.asarray(visit_data, dtype=float),
        "subject_ids": subject_ids,
        "gt_subtypes": gt_subtypes,
        "gt_baseline_stages": baseline,
        "gt_subject_stages": subj_stages,
        "gt_sequences": np.asarray(gt_sequences),
        "Z_vals": Z_vals,
        "Z_max": Z_max,
        "labels": [f"BM{i}" for i in range(n_biomarkers)],
        "N_stages": N_stages,
        "n_subjects": n_subjects,
        "n_visits": n_visits,
    }


def split_subjects_into_centres(subject_ids, n_centres=10, seed=0):
    """Split *subjects* (not visit rows) across centres; return list of visit-row
    index arrays per centre (keeping each subject's visits together)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject_ids)
    perm = rng.permutation(subjects)
    chunks = np.array_split(perm, n_centres)
    out = []
    for chunk in chunks:
        mask = np.isin(subject_ids, chunk)
        out.append(np.where(mask)[0])
    return out


def split_into_centres(n_samples, n_centres=10, seed=0, group=None, concentration=None):
    """Return a list of index arrays, one per centre.

    If ``group`` (per-subject subtype labels) and ``concentration`` are given,
    centres get heterogeneous subtype prevalence (Dirichlet-weighted sampling)
    to stress-test the federation; otherwise subjects are split uniformly.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    if group is None or concentration is None:
        return [np.sort(sh) for sh in np.array_split(idx, n_centres)]

    group = np.asarray(group)
    classes = np.unique(group)
    by_class = {c: rng.permutation(np.where(group == c)[0]).tolist() for c in classes}
    weights = rng.dirichlet(np.full(len(classes), concentration), size=n_centres)
    shards = [[] for _ in range(n_centres)]
    per_centre = n_samples // n_centres
    for ci in range(n_centres):
        for k, c in enumerate(classes):
            take = int(round(per_centre * weights[ci, k]))
            for _ in range(take):
                if by_class[c]:
                    shards[ci].append(by_class[c].pop())
    # scatter any leftovers
    leftovers = [i for c in classes for i in by_class[c]]
    for j, i in enumerate(leftovers):
        shards[j % n_centres].append(i)
    return [np.sort(np.array(sh, dtype=int)) for sh in shards]
