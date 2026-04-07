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
    """Simulate student problem responses under simple BKT model.

    Generates synthetic educational dataset by sampling problem responses
    from a Bayesian Knowledge Tracing model with fixed parameters.

    Parameters
    ----------
    nStudents : int, default 10
        Number of students to simulate.
    nProblems : int, default 20
        Number of problems to simulate.
    nKcs : int, default 1
        Number of knowledge components (KCs).
    prior : float or array-like, default 0.1
        Initial knowledge probability. Scalar broadcasted to all KCs or array of length nKcs.
    learn : float or array-like, default 0.01
        Learning (mastery) probability. Scalar or array of length nKcs.
    forget : float or array-like, default 0.05
        Forgetting probability. Scalar or array of length nKcs.
    guess : float or array-like, default 0.2
        Guessing probability (correct response without knowledge). Scalar or array of length nKcs.
    slip : float or array-like, default 0.1
        Slipping probability (incorrect response despite knowledge). Scalar or array of length nKcs.
    rng_seed : int or None, optional
        Random seed for reproducibility.
    kc_sequence : array-like of int or None, optional
        KC assignment for each problem. If None, randomly sampled.

    Returns
    -------
    pd.DataFrame
        Simulated dataset with columns: student_id, problem_id, correct, kc_id.

    Raises
    ------
    ValueError
        If parameter lengths do not match nKcs or if kc_sequence is invalid.
    """

    rng = np.random.default_rng(rng_seed)

    def _param_to_vec(x, name):
        """Convert parameter to vector format.

        Ensures parameter is a 1D array of length nKcs, broadcasting scalar
        values or validating array length.

        Parameters
        ----------
        x : float or array-like
            Input parameter value(s).
        name : str
            Parameter name for error messages.

        Returns
        -------
        np.ndarray
            1D array of shape (nKcs,).

        Raises
        ------
        ValueError
            If array size does not equal nKcs.
        """
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
