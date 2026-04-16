import pandas as pd
import numpy as np
from natsort import natsort_keygen
from typing import Sequence


def sim_simple_BKT(
    n_students: int = 10,
    n_problems: int = 20,
    n_kcs: int = 1,
    prior: float | Sequence[float] = 0.1,
    learn: float | Sequence[float] = 0.01,
    forget: float | Sequence[float] = 0.05,
    guess: float | Sequence[float] = 0.2,
    slip: float | Sequence[float] = 0.1,
    rng_seed=None,
    kc_sequence=None,
    frac=1.0,
) -> pd.DataFrame:
    """Simulate student problem responses under simple BKT model.

    Generates synthetic dataset by sampling problem responses
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
    frac : float, default 1.0
        Fraction of rows to include in the output dataset. This simulates missing data,
        or students not completing all problems, by randomly dropping rows after simulation.


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
        if arr.ndim == 0:
            arr = np.repeat(arr, n_kcs)
        else:
            arr = arr.reshape(-1)

        if arr.shape[0] != n_kcs:
            raise ValueError(f"{name} must be scalar or length n_kcs")
        return arr

    prior_vec = _param_to_vec(prior, "prior")
    learn_vec = _param_to_vec(learn, "learn")
    forget_vec = _param_to_vec(forget, "forget")
    guess_vec = _param_to_vec(guess, "guess")
    slip_vec = _param_to_vec(slip, "slip")

    if kc_sequence is None:
        kc_sequence = rng.integers(0, n_kcs, size=n_problems)
    else:
        kc_sequence = np.asarray(kc_sequence, dtype=int)
        if kc_sequence.shape[0] != n_problems:
            raise ValueError("kc_sequence must have length n_problems")
        if kc_sequence.min() < 0 or kc_sequence.max() >= n_kcs:
            raise ValueError("kc_sequence entries must be in [0, n_kcs-1]")

    knowledge = rng.random(size=(n_students, n_kcs)) < prior_vec
    correctness = np.zeros((n_students, n_problems), dtype=int)
    states = np.zeros((n_students, n_problems), dtype=int)

    for t in range(n_problems):
        kc = kc_sequence[t]
        for s in range(n_students):
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
            "timestamp": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(problem_idx.ravel(), unit="m"),
            "kc_id": "kc_" + kc_sequence[problem_idx.ravel()].astype(str),
        }
    )

    if frac < 1.0:
        data_df = data_df.sample(frac=frac, random_state=rng_seed).reset_index(
            drop=True
        )

        data_df = data_df.sort_values(
            ["student_id", "timestamp"],
            key=natsort_keygen(),  # ty:ignore[invalid-argument-type]
        ).reset_index(drop=True)
    return data_df


def sim_grouped_BKT(
    n_students: int = 10,
    n_problems: int = 20,
    n_kcs: int = 1,
    n_groups: int = 2,
    prior: float | Sequence[float] | Sequence[Sequence[float]] = 0.1,
    learn: float | Sequence[float] | Sequence[Sequence[float]] = 0.01,
    forget: float | Sequence[float] | Sequence[Sequence[float]] = 0.05,
    guess: float | Sequence[float] | Sequence[Sequence[float]] = 0.2,
    slip: float | Sequence[float] | Sequence[Sequence[float]] = 0.1,
    rng_seed=None,
    kc_sequence=None,
    group_sequence=None,
    frac=1.0,
) -> pd.DataFrame:
    """Simulate student problem responses under grouped BKT model.

    Generates synthetic dataset by sampling problem responses
    from a Bayesian Knowledge Tracing model where BKT parameters can vary by
    student group and by knowledge component (KC).

    Parameters
    ----------
    n_students : int, default 10
        Number of students to simulate.
    n_problems : int, default 20
        Number of problems to simulate.
    n_kcs : int, default 1
        Number of knowledge components (KCs).
    n_groups : int, default 2
        Number of student groups with distinct BKT parameters.
    prior : float or array-like, default 0.1
        Initial knowledge probability. Accepted formats are scalar,
        shape (n_groups,), or shape (n_groups, n_kcs).
    learn : float or array-like, default 0.01
        Learning (mastery) probability. Accepted formats are scalar,
        shape (n_groups,), or shape (n_groups, n_kcs).
    forget : float or array-like, default 0.05
        Forgetting probability. Accepted formats are scalar,
        shape (n_groups,), or shape (n_groups, n_kcs).
    guess : float or array-like, default 0.2
        Guessing probability (correct response without knowledge).
        Accepted formats are scalar, shape (n_groups,), or shape (n_groups, n_kcs).
    slip : float or array-like, default 0.1
        Slipping probability (incorrect response despite knowledge).
        Accepted formats are scalar, shape (n_groups,), or shape (n_groups, n_kcs).
    rng_seed : int or None, optional
        Random seed for reproducibility.
    kc_sequence : array-like of int or None, optional
        KC assignment for each problem. If None, randomly sampled.
    group_sequence : array-like of int or None, optional
        Group assignment for each student (0-indexed). If None, students are evenly distributed across groups.
    frac : float, default 1.0
        Fraction of rows to include in the output dataset. This simulates missing data,
        or students not completing all problems, by randomly dropping rows after simulation.

    Returns
    -------
    pd.DataFrame
        Simulated dataset with columns: student_id, problem_id, correct, kc_id, group_id, timestamp.

    Raises
    ------
    ValueError
        If parameter shapes are invalid, if kc_sequence is invalid, or if
        group_sequence is invalid.

    Notes
    -----
    Parameters can be specified per-group by providing lists/arrays of length
    n_groups, or per-(group, KC) by providing a 2D array with shape
    (n_groups, n_kcs). For example::

        sim_grouped_BKT(
            n_students=20,
            n_groups=2,
            prior=[[0.2, 0.1], [0.5, 0.4]],  # rows=groups, cols=KCs
            learn=[[0.01, 0.02], [0.05, 0.06]],
        )

    Each student is assigned a group, and knowledge states are tracked
    independently per (student, KC). Transition and emission probabilities
    are chosen from that student's group and the active KC.
    """

    rng = np.random.default_rng(rng_seed)

    def _param_to_group_kc_matrix(x, name):
        """Convert parameter to (n_groups, n_kcs) matrix.

        Supported input formats:
        - scalar: broadcast to all groups and KCs
        - 1D of length n_groups: per-group value broadcast across KCs
        - 2D of shape (n_groups, n_kcs): full per-(group, KC) specification

        Parameters
        ----------
        x : float or array-like
            Input parameter value(s).
        name : str
            Parameter name for error messages.
        Returns
        -------
        np.ndarray
            2D array of shape (n_groups, n_kcs).

        Raises
        ------
        ValueError
            If array shape is unsupported.
        """
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0:
            return np.full((n_groups, n_kcs), float(arr), dtype=float)
        if arr.ndim == 1:
            if arr.shape[0] != n_groups:
                raise ValueError(
                    f"{name} must be scalar, shape (n_groups,), or shape "
                    f"(n_groups, n_kcs); got 1D length {arr.shape[0]}"
                )
            return np.repeat(arr[:, np.newaxis], n_kcs, axis=1)
        if arr.ndim == 2 and arr.shape == (n_groups, n_kcs):
            return arr
        raise ValueError(
            f"{name} must be scalar, shape (n_groups,), or shape "
            f"(n_groups, n_kcs); got shape {arr.shape}"
        )

    prior_mat = _param_to_group_kc_matrix(prior, "prior")
    learn_mat = _param_to_group_kc_matrix(learn, "learn")
    forget_mat = _param_to_group_kc_matrix(forget, "forget")
    guess_mat = _param_to_group_kc_matrix(guess, "guess")
    slip_mat = _param_to_group_kc_matrix(slip, "slip")

    if kc_sequence is None:
        kc_sequence = rng.integers(0, n_kcs, size=n_problems)
    else:
        kc_sequence = np.asarray(kc_sequence, dtype=int)
        if kc_sequence.shape[0] != n_problems:
            raise ValueError("kc_sequence must have length n_problems")
        if kc_sequence.min() < 0 or kc_sequence.max() >= n_kcs:
            raise ValueError("kc_sequence entries must be in [0, n_kcs-1]")

    # handle group assignments for students
    if group_sequence is None:
        # evenly distribute students across groups
        group_sequence = np.zeros(n_students, dtype=int)
        for i in range(n_students):
            group_sequence[i] = i % n_groups
    else:
        group_sequence = np.asarray(group_sequence, dtype=int)
        if group_sequence.shape[0] != n_students:
            raise ValueError("group_sequence must have length n_students")
        if group_sequence.min() < 0 or group_sequence.max() >= n_groups:
            raise ValueError(
                f"group_sequence entries must be in [0, n_groups-1] ({n_groups-1})"
            )

    # each student starts with group- and KC-specific prior knowledge probability
    knowledge = np.zeros((n_students, n_kcs), dtype=bool)
    for s in range(n_students):
        group_idx = group_sequence[s]
        probs = rng.random(size=n_kcs) < prior_mat[group_idx, :]
        knowledge[s, :] = probs

    correctness = np.zeros((n_students, n_problems), dtype=int)

    # simulate BKT per problem
    for t in range(n_problems):
        kc = kc_sequence[t]
        for s in range(n_students):
            group_idx = group_sequence[s]
            knows_before = knowledge[s, kc]

            # Generate correctness based on knowledge state and group parameters
            if knows_before:
                correct = int(rng.random() >= slip_mat[group_idx, kc])
            else:
                correct = int(rng.random() < guess_mat[group_idx, kc])

            correctness[s, t] = correct

            # update knowledge state based on group-specific learning parameters
            if knows_before:
                knowledge[s, kc] = rng.random() >= forget_mat[group_idx, kc]
            else:
                knowledge[s, kc] = rng.random() < learn_mat[group_idx, kc]

    student_idx, problem_idx = np.indices(correctness.shape)

    data_df = pd.DataFrame(
        {
            "student_id": "stu_" + student_idx.ravel().astype(str),
            "problem_id": "prob_" + problem_idx.ravel().astype(str),
            "correct": correctness.ravel().astype(np.int8),
            "timestamp": pd.Timestamp("2024-01-01")
            + pd.to_timedelta(problem_idx.ravel(), unit="m"),
            "kc_id": "kc_" + kc_sequence[problem_idx.ravel()].astype(str),
            "group_id": "group_" + group_sequence[student_idx.ravel()].astype(str),
        }
    )

    if frac < 1.0:
        data_df = data_df.sample(frac=frac, random_state=rng_seed).reset_index(
            drop=True
        )

        data_df = data_df.sort_values(
            ["student_id", "timestamp"],
            key=natsort_keygen(),  # ty:ignore[invalid-argument-type]
        ).reset_index(drop=True)

    return data_df
