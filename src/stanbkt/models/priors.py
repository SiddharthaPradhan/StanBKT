"""BKT prior definitions and default prior helpers."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Union

from stanbkt.models.model_types import ModelType, PriorEstimationType


class BayesianPriors(str, Enum):
    PI_KNOW_MU = "pi_know_mu"
    PI_KNOW_STD = "pi_know_std"
    LEARN_MU = "learn_mu"
    LEARN_STD = "learn_std"
    FORGET_MU = "forget_mu"
    FORGET_STD = "forget_std"
    GUESS_MU = "guess_mu"
    GUESS_STD = "guess_std"
    SLIP_MU = "slip_mu"
    SLIP_STD = "slip_std"

    @staticmethod
    def _default_scalar_priors() -> dict["BayesianPriors", float]:
        return {
            BayesianPriors.PI_KNOW_MU: -2.0,
            BayesianPriors.PI_KNOW_STD: 5.0,
            BayesianPriors.LEARN_MU: 0.0,
            BayesianPriors.LEARN_STD: 5.0,
            BayesianPriors.FORGET_MU: -2.0,
            BayesianPriors.FORGET_STD: 5.0,
            BayesianPriors.GUESS_MU: -1.0,
            BayesianPriors.GUESS_STD: 5.0,
            BayesianPriors.SLIP_MU: -1.0,
            BayesianPriors.SLIP_STD: 5.0,
        }

    @staticmethod
    def _expand_grouped_priors(
        scalar_priors: dict["BayesianPriors", float],
        n_groups: int,
    ) -> dict["BayesianPriors", list[float]]:
        return {prior: [value] * n_groups for prior, value in scalar_priors.items()}

    @staticmethod
    def get_default_priors(
        model_type: ModelType,
        estimation_type: PriorEstimationType,
        n_groups: Optional[int] = None,
    ) -> Union[
        dict["BayesianPriors", float],
        dict["BayesianPriors", list[float]],
    ]:
        """Return default priors used for BKT parameters.

        Notes
        -----
        Priors are modeled as Normal distributions with means and standard deviations
        specified on the logit scale for probability parameters.
        """
        if estimation_type == PriorEstimationType.JOINT:
            raise NotImplementedError(
                "Joint prior estimation defaults are not implemented yet"
            )

        if estimation_type != PriorEstimationType.DEFAULT:
            raise ValueError(f"Unsupported prior estimation type: {estimation_type}")

        scalar_priors = BayesianPriors._default_scalar_priors()

        if model_type in [ModelType.STANDARD, ModelType.NESTED]:
            return scalar_priors

        if model_type == ModelType.GROUPED:
            if not isinstance(n_groups, int):
                raise ValueError(
                    "n_groups must be an integer for default grouped model priors"
                )
            if n_groups <= 0:
                raise ValueError("n_groups must be > 0 for grouped model priors")
            return BayesianPriors._expand_grouped_priors(scalar_priors, n_groups)

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def add_missing_priors(
        values: dict[BayesianPriors, float | list[float]],
        model_type: ModelType,
        estimation_type: PriorEstimationType,
        n_groups: Optional[int] = None,
    ) -> Union[
        dict["BayesianPriors", float],
        dict["BayesianPriors", list[float]],
    ]:
        """Fill missing priors with defaults and return a normalized dictionary.

        Parameters
        ----------
        values : dict[BayesianPriors | str, float | list[float]]
            Partial prior values keyed by `BayesianPriors` or by string names.
            Missing keys are filled from defaults.
        model_type : ModelType
            BKT model type for selecting prior structure.
        estimation_type : PriorEstimationType
            Prior estimation mode.
        n_groups : int, optional
            Number of groups for grouped models.
        """
        defaults = dict(
            BayesianPriors.get_default_priors(
                model_type=model_type,
                estimation_type=estimation_type,
                n_groups=n_groups,
            )
        )

        normalized_values: dict[BayesianPriors, float | list[float]] = {}
        for key, value in values.items():
            if isinstance(key, BayesianPriors):
                prior_key = key
            elif isinstance(key, str):
                try:
                    prior_key = BayesianPriors(key)
                except ValueError as exc:
                    raise ValueError(f"Unsupported prior key: {key}") from exc
            else:
                raise ValueError(f"Unsupported prior key type: {type(key).__name__}")

            normalized_values[prior_key] = value

        defaults.update(normalized_values)
        return defaults
