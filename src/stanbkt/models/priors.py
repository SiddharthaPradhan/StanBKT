"""BKT prior definitions and default prior helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union

from stanbkt.models.model_types import ModelType, PriorEstimationType


@dataclass
class BayesianPriors:
    # TODO add more for the more complex models.
    # Need to add verification for compatibility with the model type.
    # This does not depend on the estimation method as the same priors are use across
    # but it varies by the selected model type.

    pi_know_mu: float | list[float | None] | None = -2.0
    pi_know_std: float | list[float | None] | None = 5.0
    learn_mu: float | list[float | None] | None = 0.0
    learn_std: float | list[float | None] | None = 5.0
    forget_mu: float | list[float | None] | None = -2.0
    forget_std: float | list[float | None] | None = 5.0
    guess_mu: float | list[float | None] | None = -1.0
    guess_std: float | list[float | None] | None = 5.0
    slip_mu: float | list[float | None] | None = -1.0
    slip_std: float | list[float | None] | None = 5.0

    @classmethod
    def key_names(cls) -> tuple[str, ...]:
        """Return all valid prior key names in declaration order."""
        return tuple(field.name for field in fields(cls))

    def to_dict(self) -> dict[str, float | list[float | None] | None]:
        """Serialize priors to a dictionary."""
        return asdict(self)

    @staticmethod
    def _default_scalar_priors() -> dict[str, float | None]:
        return {
            key: value
            for key, value in BayesianPriors().to_dict().items()
            if isinstance(value, float) or value is None
        }

    @staticmethod
    def _expand_grouped_priors(
        scalar_priors: dict[str, float | None],
        n_groups: int,
    ) -> dict[str, list[float | None]]:
        return {prior: [value] * n_groups for prior, value in scalar_priors.items()}

    @staticmethod
    def _none_priors(
        model_type: ModelType,
        n_groups: Optional[int] = None,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        scalar_none_priors = {key: None for key in BayesianPriors.key_names()}

        if model_type in [ModelType.STANDARD, ModelType.NESTED]:
            return scalar_none_priors

        if model_type == ModelType.GROUPED:
            if not isinstance(n_groups, int):
                raise ValueError("n_groups must be an integer for grouped model priors")
            if n_groups <= 0:
                raise ValueError("n_groups must be > 0 for grouped model priors")
            return BayesianPriors._expand_grouped_priors(scalar_none_priors, n_groups)

        raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def get_default_priors(
        model_type: ModelType,
        estimation_type: PriorEstimationType,
        n_groups: Optional[int] = None,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
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
        values: dict[str, Any],
        model_type: ModelType,
        estimation_type: PriorEstimationType,
        n_groups: Optional[int] = None,
        defaults: bool = True,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        """Fill missing priors and return a normalized dictionary.

        Parameters
        ----------
        values : dict[str, float | list[float | None]]
            Partial prior values keyed by prior key string names.
            Missing keys are filled from defaults or `None` based on `defaults`.
        model_type : ModelType
            BKT model type for selecting prior structure.
        estimation_type : PriorEstimationType
            Prior estimation mode.
        n_groups : int, optional
            Number of groups for grouped models.
        defaults : bool, optional
            When ``True``, missing keys are filled with default prior values.
            When ``False``, missing keys are filled with ``None`` (or ``[None] * n_groups``
            for grouped models).
        """
        base_priors = dict(
            BayesianPriors.get_default_priors(
                model_type=model_type,
                estimation_type=estimation_type,
                n_groups=n_groups,
            )
            if defaults
            else BayesianPriors._none_priors(model_type=model_type, n_groups=n_groups)
        )

        valid_keys = set(BayesianPriors.key_names())
        normalized_values: dict[str, float | list[float | None] | None] = {}
        for key, value in values.items():
            if not isinstance(key, str):
                raise ValueError(f"Unsupported prior key type: {type(key).__name__}")
            if key not in valid_keys:
                raise ValueError(f"Unsupported prior key: {key}")
            prior_key = key

            normalized_values[prior_key] = value

        base_priors.update(normalized_values)
        return base_priors
