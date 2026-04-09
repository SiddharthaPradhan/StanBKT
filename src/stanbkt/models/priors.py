"""BKT prior definitions and default prior helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union, TYPE_CHECKING

from stanbkt.models.model_types import ModelType, InitKnowledgeStrategy

if TYPE_CHECKING:
    from stanbkt.models.core.base import BKTModelBase
    from stanbkt.models.core.standard import StandardBKT


class _UnsetType:
    """Empty class used as a sentinel type for unset parameters."""

    pass


_UNSET = _UnsetType()  # sentinel value for unset parameters


#  inherit this for non-standard models and customize behavior as needed.
@dataclass
class BayesianPriors:
    """Bayesian priors for BKT model parameters.

    Stores prior specifications (means and standard deviations on logit scale)
    for the four core BKT parameters: prior knowledge (pi_know), learning rate
    (learn), forgetting rate (forget), guessing (guess), and slipping (slip).

    Exclicitly passing None for a parameter indicates an improper and non-informative prior (Stan uses a uniform prior).
    If all parameters are left as None, the model will use non-informative priors for all parameters, which implies
    the MAP solution will be equivalent to maximum likelihood estimation.

    Each parameter can be specified as a scalar (standard model), a list of
    values (grouped model), or None (non-informative priors).

    Guess and slip are modeled on the half-logit scale to constrain them to [0, 0.5].
    All other parameters are modeled on the logit scale.

    Parameters
    ----------
    pi_know_mu : float | list[float | None] | None, default -2.0
        Prior mean for logit-scale initial knowledge probability.
    pi_know_std : float | list[float | None] | None, default 5.0
        Prior std dev for logit-scale initial knowledge probability.
    learn_mu : float | list[float | None] | None, default 0.0
        Prior mean for logit-scale learning rate.
    learn_std : float | list[float | None] | None, default 5.0
        Prior std dev for logit-scale learning rate.
    forget_mu : float | list[float | None] | None, default -2.0
        Prior mean for logit-scale forgetting rate.
    forget_std : float | list[float | None] | None, default 5.0
        Prior std dev for logit-scale forgetting rate.
    guess_mu : float | list[float | None] | None, default -1.0
        Prior mean for logit-scale guessing probability.
    guess_std : float | list[float | None] | None, default 5.0
        Prior std dev for logit-scale guessing probability.
    slip_mu : float | list[float | None] | None, default -1.0
        Prior mean for logit-scale slipping probability.
    slip_std : float | list[float | None] | None, default 5.0
        Prior std dev for logit-scale slipping probability.
    """

    # TODO add more for the more complex models.
    # Need to add verification for compatibility with the model type.
    # This does not depend on the estimation method as the same priors are used
    # across methods, but it varies by the selected model type.

    pi_know_mu: float | list[float | None] | None | _UnsetType = _UNSET
    pi_know_std: float | list[float | None] | None | _UnsetType = _UNSET
    learn_mu: float | list[float | None] | None | _UnsetType = _UNSET
    learn_std: float | list[float | None] | None | _UnsetType = _UNSET
    forget_mu: float | list[float | None] | None | _UnsetType = _UNSET
    forget_std: float | list[float | None] | None | _UnsetType = _UNSET
    guess_mu: float | list[float | None] | None | _UnsetType = _UNSET
    guess_std: float | list[float | None] | None | _UnsetType = _UNSET
    slip_mu: float | list[float | None] | None | _UnsetType = _UNSET
    slip_std: float | list[float | None] | None | _UnsetType = _UNSET
    use_defaults: bool = True

    def __post_init__(self) -> None:
        """Post-initialization processing to handle default values."""
        if self.use_defaults:
            defaults = BayesianPriors._get_default_priors(
                model_type=ModelType.STANDARD,  # default structure for defaults
                estimation_type=InitKnowledgeStrategy.CORRECTNESS_ONLY,  # default estimation type for defaults
            )
            for key, default_value in defaults.items():
                if getattr(self, key) is _UNSET:
                    setattr(self, key, default_value)
        else:
            # If not using defaults, set any unset fields to None
            for field in fields(self):
                if getattr(self, field.name) is _UNSET:
                    setattr(self, field.name, None)

    @classmethod
    def key_names(cls) -> tuple[str, ...]:
        """Return all valid prior key names in declaration order."""
        return tuple(field.name for field in fields(cls))

    def to_dict(self) -> dict[str, float | list[float | None] | None]:
        """Serialize priors to a dictionary."""
        return asdict(self)

    @staticmethod
    def _validate(
        priors: Union[BayesianPriors, dict[str, BayesianPriors]],
        model_class: type[BKTModelBase],
    ) -> None:
        if isinstance(priors, BayesianPriors):
            BayesianPriors._validate_single(priors, model_class)
        elif isinstance(priors, dict):
            for kc, kc_priors in priors.items():
                if not isinstance(kc_priors, BayesianPriors):
                    raise ValueError(
                        f"Invalid prior specification for KC '{kc}': expected BayesianPriors instance, got {type(kc_priors).__name__}"
                    )
                BayesianPriors._validate_single(kc_priors, model_class, kc_id=kc)

    @staticmethod
    def _validate_single(
        single_priors: BayesianPriors,
        model_class: type[BKTModelBase],
        kc_id: str | None = None,
    ) -> None:
        """Validate that the prior values are compatible with expected types and model structure.

        Checks that each prior value is either a float, a list of floats/None, or None.
        For grouped models, checks that list lengths match the expected number of groups.
        """
        # TODO add more as you go
        MU = "mu"
        STD = "std"
        for key, value in single_priors.to_dict().items():
            # check type for model class type
            if issubclass(model_class, StandardBKT):
                if isinstance(value, list):
                    raise ValueError(
                        f"Invalid prior value type for {key} in standard model: expected float or None, got list"
                    )

            # TODO: for grouped models, check that lists have the correct length
            if not isinstance(value, (float, list, type(None))):
                raise ValueError(
                    f"Invalid prior value type for {key}: expected float, list, or None, got {type(value).__name__}"
                )
            # check std is positive if not None
            if key.endswith(STD) and value is not None:
                if isinstance(value, float):
                    if value <= 0:
                        msg = f"Bayesian Prior {key} "
                        if kc_id is not None:
                            msg += f"for KC '{kc_id}' "
                        msg += f"must be positive, got {value}"
                        raise ValueError(msg)
                elif isinstance(value, list):
                    for i, v in enumerate(value):
                        if v is not None and v <= 0:
                            msg = f"Bayesian Prior {key}[{i}] "
                            if kc_id is not None:
                                msg += f"for KC '{kc_id}' "
                            msg += f"must be positive, got {v}"
                            raise ValueError(msg)

    @staticmethod
    def _default_scalar_priors() -> dict[str, float | None]:
        """Extract scalar defaults from a standard BayesianPriors instance.

        Returns
        -------
        dict[str, float | None]
            Mapping of prior keys to their default scalar values.
        """

        return {
            "pi_know_mu": -2.0,
            "pi_know_std": 5.0,
            "learn_mu": 0.0,
            "learn_std": 5.0,
            "forget_mu": -2.0,
            "forget_std": 5.0,
            "guess_mu": -1.0,
            "guess_std": 5.0,
            "slip_mu": -1.0,
            "slip_std": 5.0,
        }

    @staticmethod
    def _expand_grouped_priors(
        scalar_priors: dict[str, float | None],
        n_groups: int,
    ) -> dict[str, list[float | None]]:
        """Expand scalar priors to grouped model format.

        Replicates each scalar prior value ``n_groups`` times to create
        parameter lists suitable for grouped BKT models.

        Parameters
        ----------
        scalar_priors : dict[str, float | None]
            Mapping of prior names to scalar values.
        n_groups : int
            Number of groups (replicas) for each prior value.

        Returns
        -------
        dict[str, list[float | None]]
            Mapping of prior names to lists of replicated prior values.
        """
        return {prior: [value] * n_groups for prior, value in scalar_priors.items()}

    @staticmethod
    def _none_priors(
        model_type: ModelType,
        n_groups: Optional[int] = None,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        """Return priors with all values set to None (improper and non-informative).
        Stan uses a uniform prior.

        Parameters
        ----------
        model_type : ModelType
            The model type (STANDARD, GROUPED, or NESTED).
        n_groups : Optional[int]
            Number of groups (required for GROUPED model type, ignored for others).

        Returns
        -------
        Union[dict[str, float | None], dict[str, list[float | None]]]
            Mapping of prior names to None values, structured according to model_type.

        Raises
        ------
        ValueError
            If model_type is GROUPED and n_groups is not a positive integer.
        """
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
    def _get_default_priors(
        model_type: ModelType,
        estimation_type: InitKnowledgeStrategy,
        n_groups: Optional[int] = None,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        """Return default priors used for BKT parameters.

        Notes
        -----
        Priors are modeled as Normal distributions with means and standard
        deviations specified on the logit scale for learn and forget probability parameters.
        Guess and slip are modeled on the half-logit scale to constrain them to [0, 0.5].
        """
        if estimation_type not in [
            InitKnowledgeStrategy.CORRECTNESS_ONLY,
            InitKnowledgeStrategy.JOINT,
        ]:
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
    def _add_missing_priors(
        values: dict[str, Any],
        model_type: ModelType,
        estimation_type: InitKnowledgeStrategy,
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
            Missing keys are filled from defaults or ``None`` based on ``defaults``.
        model_type : ModelType
            BKT model type for selecting prior structure.
        estimation_type : InitKnowledgeStrategy
            Prior estimation mode.
        n_groups : int, optional
            Number of groups for grouped models.
        defaults : bool, optional
            When ``True``, missing keys are filled with default prior values.
            When ``False``, missing keys are filled with ``None`` (or
            ``[None] * n_groups`` for grouped models).
        """
        base_priors = dict(
            BayesianPriors._get_default_priors(
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
            normalized_values[key] = value

        base_priors.update(normalized_values)
        return base_priors


@dataclass
class BayesianPriorsJoint(BayesianPriors):
    """Bayesian priors for joint estimation of initial knowledge and other parameters.

    Adds b0 and b1 parameters for modeling the linear relationship between initial knowledge logit
    and additional data used in the `InitKnowledgeStrategy.JOINT` estimation strategy.
    """

    beta0_mu: float | None | _UnsetType = _UNSET
    beta0_std: float | None | _UnsetType = _UNSET

    beta1_mu: float | None | _UnsetType = _UNSET
    beta1_std: float | None | _UnsetType = _UNSET
