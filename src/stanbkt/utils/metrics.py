import numpy as np
import numpy.typing as npt


def accuracy(correctness: npt.ArrayLike, predictions: npt.ArrayLike) -> float:
    """Compute accuracy by thresholding predicted probabilities at 0.5.

    Parameters
    ----------
    correctness : array-like
        Flattened binary correctness labels (0 or 1).
    predictions : array-like
        Flattened predicted probabilities in [0, 1].

    Returns
    -------
    float
        Proportion of correctly classified observations.
    """
    correctness = np.asarray(correctness)
    predictions = np.asarray(predictions)
    predicted_labels = (predictions >= 0.5).astype(int)
    return float(np.mean(predicted_labels == correctness))


def rmse(correctness: npt.ArrayLike, predictions: npt.ArrayLike) -> float:
    """Compute root mean squared error between predicted probabilities and binary labels.

    Parameters
    ----------
    correctness : array-like
        Flattened binary correctness labels (0 or 1).
    predictions : array-like
        Flattened predicted probabilities in [0, 1].

    Returns
    -------
    float
        Root mean squared error.
    """
    correctness = np.asarray(correctness, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    return float(np.sqrt(np.mean((correctness - predictions) ** 2)))


def auc(correctness: npt.ArrayLike, predictions: npt.ArrayLike) -> float:
    """Compute the area under the ROC curve (AUC).


    Parameters
    ----------
    correctness : array-like
        Flattened binary correctness labels (0 or 1).
    predictions : array-like
        Flattened predicted probabilities in [0, 1].

    Returns
    -------
    float
        AUC score in [0, 1].
    """
    correctness = np.asarray(correctness)
    predictions = np.asarray(predictions)

    # sort by descending predicted probability
    desc_order = np.argsort(-predictions)
    sorted_labels = correctness[desc_order]

    n_pos = int(np.sum(sorted_labels == 1))
    n_neg = int(np.sum(sorted_labels == 0))

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "AUC is undefined when only one class is present in correctness."
        )

    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapezoid(tpr, fpr))
