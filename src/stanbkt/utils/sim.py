import pandas as pd
import numpy as np


def sim_simple_BKT(
    nStudents: int = 10,
    nProblems: int = 20,
    nKcs: int = 1,
    prior=0.1,
    learn=0.01,
    forget=0.05,
    guess=0.2,
    slip=0.1,
    rng_seed=None,
    kc_sequence=None,
) -> pd.DataFrame:

    rng = np.random.default_rng(rng_seed)

    def _param_to_vec(x, name):
        arr = np.asarray(x, dtype=float)
        if arr.size == 1:
            arr = np.repeat(arr, nKcs)
        if arr.size != nKcs:
            raise ValueError(f"{name} must be scalar or length nKcs")
        return arr

    prior_vec = _param_to_vec(prior, "prior")
    learn_vec = _param_to_vec(learn, "learn")
    forget_vec = _param_to_vec(forget, "forget")
    guess_vec = _param_to_vec(guess, "guess")
    slip_vec = _param_to_vec(slip, "slip")

    if kc_sequence is None:
        kc_sequence = rng.integers(0, nKcs, size=nProblems)
    else:
        kc_sequence = np.asarray(kc_sequence, dtype=int)
        if kc_sequence.shape[0] != nProblems:
            raise ValueError("kc_sequence must have length nProblems")
        if kc_sequence.min() < 0 or kc_sequence.max() >= nKcs:
            raise ValueError("kc_sequence entries must be in [0, nKcs-1]")

    knowledge = rng.random(size=(nStudents, nKcs)) < prior_vec
    correctness = np.zeros((nStudents, nProblems), dtype=int)
    states = np.zeros((nStudents, nProblems), dtype=int)

    for t in range(nProblems):
        kc = kc_sequence[t]
        for s in range(nStudents):
            knows_before = knowledge[s, kc]
            if knows_before:
                correct = int(rng.random() >= slip_vec[kc])
            else:
                correct = int(rng.random() < guess_vec[kc])

            correctness[s, t] = correct

            if knows_before:
                knowledge[s, kc] = rng.random() >= forget_vec[kc]
            else:
                knowledge[s, kc] = rng.random() < learn_vec[kc]

            states[s, t] = knowledge[s, kc]

    student_idx, problem_idx = np.indices(correctness.shape)

    data_df = pd.DataFrame(
        {
            "student_id": "stu_" + student_idx.ravel().astype(str),
            "problem_id": "prob_" + problem_idx.ravel().astype(str),
            "correct": correctness.ravel().astype(np.int8),
            "kc_id": "kc_" + kc_sequence[problem_idx.ravel()].astype(str),
        }
    )

    return data_df
