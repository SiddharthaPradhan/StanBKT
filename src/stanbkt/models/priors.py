"""BKT prior definitions and default prior helpers."""

from __future__ import annotations
from abc import abstractmethod, ABC

from dataclasses import asdict, dataclass, fields
from typing import Any, Optional, Union, TYPE_CHECKING

from stanbkt.models.model_types import ModelType, InitKnowledgeStrategy

# prevents circular imports for type checking only
if TYPE_CHECKING:
    from stanbkt.models.core.base import BKTModelBase
    from stanbkt.models.core.standard import StandardBKT
    from stanbkt.models.core.multi import MultiBKT
    from stanbkt.models.core.hierarchical import HierarchicalBKT


# sentinel type and value for unset parameters
class _UnsetType:
    """Empty class used as a sentinel type for unset parameters."""

    pass


_UNSET = _UnsetType()  # sentinel value for unset parameters

# constant suffixes used in validation
STD = "std"
LAMBDA = "lambda"


# keys that relate to the InitKnowledgeStrategy.JOINT estimation strategy
# and should be excluded when using CORRECTNESS_ONLY strategy
JOINT_STRATEGY_KEYS: list[str] = [
    "pi_b0_know_mu",
    "pi_b0_know_std",
    "pi_b1_know_mu",
    "pi_b1_know_std",
    "pi_sigma_lambda",
]

# keys that relate to the InitKnowledgeStrategy.CORRECTNESS_ONLY estimation strategy
# and should be excluded when using JOINT strategy
CORRECTNESS_ONLY_STRATEGY_KEYS: list[str] = [
    "pi_know_mu",
    "pi_know_std",
]

# keys that relate to the post-test outcome regression. Unlike the JOINT keys above,
# these are independent of InitKnowledgeStrategy and are never popped in to_dict();
# _build_stan_data_dict decides whether to forward them based on post_outcome_regression.
POST_OUTCOME_STRATEGY_KEYS: list[str] = [
    "post_b0_mu",
    "post_b0_std",
    "post_b_mastery_mu",
    "post_b_mastery_std",
    "post_b1_mu",
    "post_b1_std",
    "post_sigma_lambda",
]

USE_DEFAULTS_KEY = "use_defaults"


# the following classes are better as a typed dict, but as of Python 3.14,
# typed dict do not support default values, so dataclasses are used instead.
@dataclass
class PriorsBase(ABC):
    """Base class containing common functionality for Bayesian priors used in BKT models."""

    # priors for the linear regression used in the JOINT estimation strategy
    # logit_pi_know ~ beta0 + beta1 * pretest + sigma* logit_pi_know_z
    pi_b0_know_mu: float | None | _UnsetType = _UNSET
    pi_b0_know_std: float | None | _UnsetType = _UNSET
    pi_b1_know_mu: float | list[float] | None | _UnsetType = _UNSET
    pi_b1_know_std: float | list[float] | None | _UnsetType = _UNSET
    pi_sigma_lambda: float | None | _UnsetType = _UNSET

    # priors for the post-test outcome regression, independent of InitKnowledgeStrategy
    # post_outcome ~ post_b0 + post_b_mastery * final_pknow + post_b1 * post_covariate, residual std post_sigma
    post_b0_mu: float | None | _UnsetType = _UNSET
    post_b0_std: float | None | _UnsetType = _UNSET
    post_b_mastery_mu: float | None | _UnsetType = _UNSET
    post_b_mastery_std: float | None | _UnsetType = _UNSET
    post_b1_mu: float | list[float] | None | _UnsetType = _UNSET
    post_b1_std: float | list[float] | None | _UnsetType = _UNSET
    post_sigma_lambda: float | None | _UnsetType = _UNSET

    # whether to fill in missing values with defaults or None (non-informative)
    use_defaults: bool = True

    @abstractmethod
    def __post_init__(self) -> None:
        """Post-initialization processing to handle default values."""
        raise NotImplementedError(
            "Subclasses must implement __post_init__ to handle defaults"
        )

    @staticmethod
    @abstractmethod
    def expected_class() -> type[BKTModelBase]:
        """Return the expected BKT model class type for these priors."""
        raise NotImplementedError(
            "Subclasses must implement expected_class to specify compatible model type"
        )

    def to_dict(
        self, estimation_type: InitKnowledgeStrategy
    ) -> Union[dict[str, float | None], dict[str, list[float | None]]]:
        """Serialize priors to a dictionary."""
        prior_dict = asdict(self)
        if estimation_type == InitKnowledgeStrategy.CORRECTNESS_ONLY:
            for key in JOINT_STRATEGY_KEYS:
                prior_dict.pop(key, None)

        elif estimation_type == InitKnowledgeStrategy.JOINT:
            for key in CORRECTNESS_ONLY_STRATEGY_KEYS:
                prior_dict.pop(key, None)
        else:
            raise ValueError(f"Unsupported estimation type: {estimation_type}")

        prior_dict.pop(USE_DEFAULTS_KEY, None)  # remove internal flag from output
        return prior_dict

    @classmethod
    def key_names(cls) -> tuple[str, ...]:
        """Return all valid prior key names."""
        return tuple(field.name for field in fields(cls))

    @classmethod
    def _validate(
        cls: type[PriorsBase],
        priors: Union[PriorsBase, dict[str, PriorsBase]],
        model_class: type[BKTModelBase],
        estimation_strategy: InitKnowledgeStrategy,
        n_groups: int | None = None,
    ) -> None:
        """
        Validate the provided priors. This mainly checks the lambda and std parameters to ensure they are positive and non-zero.
        If a dict is passed, it implies priors are being specified separately for each KC.
        """
        if isinstance(priors, cls):
            cls._validate_single(
                priors, model_class, estimation_strategy, n_groups=n_groups
            )
        elif isinstance(priors, PriorsBase):
            raise TypeError(
                f"Invalid prior class: expected {cls.__name__} for {model_class.__name__}, "
                f"got {type(priors).__name__}"
            )
        elif isinstance(priors, dict):
            for kc_id, kc_priors in priors.items():
                if not isinstance(kc_priors, cls):
                    raise ValueError(
                        f"Invalid prior specification for KC '{kc_id}': expected "
                        f"`{cls.__name__}` instance, got {type(kc_priors).__name__}"
                    )
                # validate the priors for this KC, KC ID is passed for informative error messages
                cls._validate_single(
                    kc_priors,
                    model_class,
                    estimation_strategy,
                    kc_id=str(kc_id),
                    n_groups=n_groups,
                )
        else:
            raise TypeError(
                f"Invalid priors argument: expected {cls.__name__} or dict[str, {cls.__name__}], "
                f"got {type(priors).__name__}"
            )

    @staticmethod
    @abstractmethod
    def _validate_single(
        single_priors: PriorsBase,
        model_class: type[BKTModelBase],
        estimation_strategy: InitKnowledgeStrategy,
        kc_id: str | None = None,
        n_groups: int | None = None,  # unused in standard priors, but left here for signature consistency
    ) -> None:
        """Validate that the prior values are compatible with expected types and model structure."""
        raise NotImplementedError(
            "Subclasses must implement _validate_single to check prior value compatibility"
        )

    @staticmethod
    def _default_scalar_priors(
        return_none=False,
    ) -> dict[str, Union[float, None]]:
        """Extract scalar defaults from a standard StandardPriors instance.

        Parameters
        ----------
        return_none : bool, default False
            If True, return a dict with all keys set to None (improper and non-informative).
            If False, return a dict with default scalar values.

        Returns
        -------
        dict[str, float | None]
            Mapping of prior keys to their default scalar values.
        """
        # need explict typing with float | None as `dict` is invariant
        default_dict: dict[str, float | None] = {
            "learn_mu": 0.0,
            "learn_std": 5.0,
            "forget_mu": -2.0,
            "forget_std": 5.0,
            "guess_mu": -1.0,
            "guess_std": 5.0,
            "slip_mu": -1.0,
            "slip_std": 5.0,
            "pi_know_mu": -2.0,
            "pi_know_std": 5.0,
            "pi_b0_know_mu": 0.0,
            "pi_b0_know_std": 5.0,
            "pi_b1_know_mu": 0.0,
            "pi_b1_know_std": 5.0,
            "pi_sigma_lambda": 5.0,  # mean 0.2, sigma and b1 are weakly identified from one initial state per student
            "post_b0_mu": 0.0,
            "post_b0_std": 5.0,
            "post_b_mastery_mu": 0.0,
            "post_b_mastery_std": 5.0,
            "post_b1_mu": 0.0,
            "post_b1_std": 5.0,
            "post_sigma_lambda": 0.5,
        }
        if not return_none:
            return default_dict
        else:
            return {key: None for key in default_dict}

    @staticmethod
    @abstractmethod
    def get_default_priors(
        estimation_type: InitKnowledgeStrategy,
        return_none: bool = False,
        n_groups: int = 0,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        raise NotImplementedError(
            "Subclasses must implement get_default_priors to provide appropriate defaults based on estimation strategy"
        )

    @staticmethod
    def _expand_covariate_priors(
        mu: float | list[float] | None,
        std: float | list[float] | None,
        n_covariates: int,
    ) -> tuple[list[float] | None, list[float] | None]:
        """Broadcast scalar pi_b1_know_mu/std priors to length ``n_covariates``.

        A list is validated against ``n_covariates`` and returned unchanged.
        This is separate from :meth:`MultiPriors._expand_grouped_priors` since
        covariates are a different axis than groups.
        """

        def _expand(value: float | list[float] | None) -> list[float] | None:
            if value is None:
                return None
            if isinstance(value, list):
                if len(value) != n_covariates:
                    raise ValueError(
                        f"pi_b1_know prior must be a list of length {n_covariates} "
                        f"(number of covariates), got list of length {len(value)}."
                    )
                return value
            return [value] * n_covariates

        return _expand(mu), _expand(std)


@dataclass
class StandardPriors(PriorsBase):
    """Bayesian priors for the Standard BKT model parameters.

    Stores prior specifications (means and standard deviations on logit scale)
    for the four core BKT parameters: prior knowledge (pi_know), learning rate
    (learn), forgetting rate (forget), guessing (guess), and slipping (slip).

    Exclicitly passing None for a parameter indicates an improper and non-informative prior
    (Stan uses a uniform prior). If all parameters are left as None, the model will use
    non-informative priors for all parameters, which implies the MAP solllution will be
    equivalent to maximum likelihood estimation.

    Each parameter can be specified as a scalar or None (non-informative prior). Use
    :class:`MultiPriors` for per group values with :class:`MultiBKT`.

    Guess and slip are modeled on the half-logit scale to constrain them to [0, 0.5].
    All other parameters are modeled on the logit scale.

    Parameters
    ----------

    learn_mu : float |  None, default 0.0
        Prior mean for logit-scale learning rate.
    learn_std : float |  None, default 5.0
        Prior std dev for logit-scale learning rate.
    forget_mu : float |  None, default -2.0
        Prior mean for logit-scale forgetting rate.
    forget_std : float |  None, default 5.0
        Prior std dev for logit-scale forgetting rate.
    guess_mu : float |  None, default -1.0
        Prior mean for half-logit-scale guessing probability.
    guess_std : float |  None, default 5.0
        Prior std dev for half-logit-scale guessing probability.
    slip_mu : float |  None, default -1.0
        Prior mean for half-logit-scale slipping probability.
    slip_std : float |  None, default 5.0
        Prior std dev for half-logit-scale slipping probability.
    pi_know_mu : float |  None, default -2.0
        Prior mean for logit-scale initial knowledge probability.
    pi_know_std : float |  None, default 5.0
        Prior std dev for logit-scale initial knowledge probability.
    pi_b0_know_mu : float | None, default 0.0
        Prior mean for the intercept (b0) in the regression of logit initial knowledge.
        Used only with `InitKnowledgeStrategy.JOINT`.
    pi_b0_know_std : float | None, default 5.0
        Prior std dev for the intercept (b0). Used only with `InitKnowledgeStrategy.JOINT`.
    pi_b1_know_mu : float | list[float] | None, default 0.0
        Prior mean for the covariate slopes (b1). A scalar applies to every covariate, a list
        gives one value per covariate. Used only with `InitKnowledgeStrategy.JOINT`.
    pi_b1_know_std : float | list[float] | None, default 5.0
        Prior std dev for the covariate slopes (b1), scalar or one value per covariate.
        Used only with `InitKnowledgeStrategy.JOINT`.
    pi_sigma_lambda : float | None, default 5.0
        Rate of the Exponential prior on the regression residual std dev (sigma), prior mean 0.2.
        Kept tight because sigma and the b1 slopes are weakly identified, a loose prior inflates
        both. Used only with `InitKnowledgeStrategy.JOINT`.
    post_b0_mu : float | None, default 0.0
        Prior mean for the intercept of the post-test outcome regression.
        Used only when `post_outcome_regression=True`.
    post_b0_std : float | None, default 5.0
        Prior std dev for the post-test outcome intercept.
    post_b_mastery_mu : float | None, default 0.0
        Prior mean for the effect of final mastery on the post-test outcome.
    post_b_mastery_std : float | None, default 5.0
        Prior std dev for the effect of final mastery on the post-test outcome.
    post_b1_mu : float | list[float] | None, default 0.0
        Prior mean for the post-test covariate slopes, scalar or one value per covariate.
    post_b1_std : float | list[float] | None, default 5.0
        Prior std dev for the post-test covariate slopes, scalar or one value per covariate.
    post_sigma_lambda : float | None, default 0.5
        Rate of the Exponential prior on the post-test outcome residual std dev.
    use_defaults : bool, default True
        Whether to fill in missing prior values with defaults (True) or None (False).
        If True, any parameter not explicitly set will be filled with a default value.
        If False, any parameter not explicitly set will be set to None, indicating a non-informative uniform prior.

    """

    # Note: any changes here should also be reflected in the default priors returned by _get_default_priors()
    learn_mu: float | None | _UnsetType = _UNSET
    learn_std: float | None | _UnsetType = _UNSET
    forget_mu: float | None | _UnsetType = _UNSET
    forget_std: float | None | _UnsetType = _UNSET
    guess_mu: float | None | _UnsetType = _UNSET
    guess_std: float | None | _UnsetType = _UNSET
    slip_mu: float | None | _UnsetType = _UNSET
    slip_std: float | None | _UnsetType = _UNSET

    pi_know_mu: float | None | _UnsetType = _UNSET
    pi_know_std: float | None | _UnsetType = _UNSET

    @staticmethod
    def expected_class() -> type[BKTModelBase]:
        """Return the expected BKT model class type for these priors."""
        from stanbkt.models.core.standard import StandardBKT

        return StandardBKT

    def __post_init__(self) -> None:
        """Post-initialization processing to handle default values."""
        if self.use_defaults:
            # unfiltered scalar defaults (covers both CORRECTNESS_ONLY and JOINT keys) -
            # filtering by estimation strategy happens later, at to_dict() call time.
            defaults = StandardPriors._default_scalar_priors()
            for key, default_value in defaults.items():
                if getattr(self, key) is _UNSET:
                    setattr(self, key, default_value)
        else:
            # If not using defaults, set any unset fields to None
            for field in fields(self):
                if getattr(self, field.name) is _UNSET:
                    setattr(self, field.name, None)

    @staticmethod
    def _validate_single(
        single_priors: PriorsBase,
        model_class: type[BKTModelBase],
        estimation_strategy: InitKnowledgeStrategy,
        kc_id: str | None = None,
        n_groups: int | None = None,  # unused in standard priors, but left here for signature consistency
    ) -> None:
        """Validate that the prior values are compatible with expected types and model structure.

        Check that each prior value is either a float or None. For std and lambda parameters,
        check that values are positive if not None.
        """

        for key, value in single_priors.to_dict(estimation_strategy).items():
            # check type for model class type
            if not issubclass(model_class, StandardPriors.expected_class()):
                raise TypeError(
                    f"Invalid model class: {StandardPriors.__name__} should be used with {StandardPriors.expected_class().__name__}, "
                    f"got {model_class.__name__}"
                )
            # pi_b1_know_mu/std and post_b1_mu/std are per-covariate and may be a list; every other key is scalar
            if key in ("pi_b1_know_mu", "pi_b1_know_std", "post_b1_mu", "post_b1_std") and isinstance(value, list):
                if not all(isinstance(v, (int, float, type(None))) for v in value):
                    raise ValueError(
                        f"Invalid prior value type for {key}: expected a list of float or None values."
                    )
            elif not isinstance(value, (int, float, type(None))):
                raise ValueError(
                    f"Invalid prior value type for {key}: expected float, or None, got {type(value).__name__}"
                )
            # check std and lambda are positive if not None
            if key.endswith((STD, LAMBDA)) and value is not None:
                values_to_check = value if isinstance(value, list) else [value]
                for v in values_to_check:
                    if v is not None and v <= 0:
                        msg = f"Bayesian Prior {key} "
                        if kc_id is not None:
                            msg += f"for KC '{kc_id}' "
                        msg += f"must be positive and non-zero, got {v}"
                        raise ValueError(msg)

    @staticmethod
    def get_default_priors(
        estimation_type: InitKnowledgeStrategy,
        return_none: bool = False,
        n_groups: int = 0,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        """Return default priors dictionary used for BKT parameters.

        Parameters
        ----------
        estimation_type : InitKnowledgeStrategy
            The strategy used for estimating initial knowledge, which determines which priors are relevant.
        return_none : bool, default False
            If True, return a dict with all keys set to None (improper and non-informative).
            If False, return a dict with default scalar values.
        n_groups : int, default 0
            The number of groups for grouped models. This is kept here for signature consistency
            but is not used for the standard priors since the parameters are scalar.

        Notes
        -----
        BKT specific priors are modeled as a Normal distributions with means and standard
        deviations specified on the logit scale for learn and forget probability parameters.
        Guess and slip are modeled on the half-logit scale to constrain them to [0, 0.5] on the probability scale.

        In cases where estimation_type = :attr:`stanbkt.models.model_types.InitKnowledgeStrategy.JOINT`, the b0 and b1 parameters
        are modeled as Normal distributions. Additionally, the
        regression residuals is modeled as a positive-valued parameter (pi_sigma_lambda) with a Exponential prior.


        A non-centered parameterization is used:

            logit_pi_know ~ pi_b0_know + pi_b1_know * pretest + pi_sigma * logit_pi_know_z

            Where,
                - pi_b0_know ~ Normal(pi_b0_know_mu, pi_b0_know_std)
                - pi_b1_know ~ Normal(pi_b1_know_mu, pi_b1_know_std)
                - pi_sigma ~ Exponential(pi_sigma_lambda)
                - logit_pi_know_z ~ N(0, 1).

        This model is mathematically equivalent to:

        logit_pi_know ~ Normal(pi_b0_know + pi_b1_know * pretest, pi_sigma)

        but the former (non-centered) parameterization is computationally more efficient and has better
        convergence properties in Stan.
        """

        if estimation_type not in [e for e in InitKnowledgeStrategy]:
            raise ValueError(f"Unsupported prior estimation type: {estimation_type}")

        scalar_priors = StandardPriors._default_scalar_priors(return_none=return_none)
        # use_defaults=False: all values are already explicitly provided, avoids recursion
        return StandardPriors(**scalar_priors, use_defaults=False).to_dict(
            estimation_type=estimation_type
        )


@dataclass
class MultiPriors(PriorsBase):
    """Bayesian priors for the grouped (MultiBKT) model parameters.

    Same fields and defaults as :class:`StandardPriors`, but the core BKT parameters
    (learn, forget, guess, slip and pi_know) also accept a list with one value per
    group. Groups are indexed per KC in order of first appearance, with students in
    natural sort order (see ``summary()`` labels). A scalar applies to every group, and
    None gives a non-informative uniform prior. The JOINT regression and post-test
    outcome priors are shared across groups.

    Parameters
    ----------
    learn_mu : float | list[float] | None, default 0.0
        Prior mean for logit-scale learning rate.
    learn_std : float | list[float] | None, default 5.0
        Prior std dev for logit-scale learning rate.
    forget_mu : float | list[float] | None, default -2.0
        Prior mean for logit-scale forgetting rate.
    forget_std : float | list[float] | None, default 5.0
        Prior std dev for logit-scale forgetting rate.
    guess_mu : float | list[float] | None, default -1.0
        Prior mean for half-logit-scale guessing probability.
    guess_std : float | list[float] | None, default 5.0
        Prior std dev for half-logit-scale guessing probability.
    slip_mu : float | list[float] | None, default -1.0
        Prior mean for half-logit-scale slipping probability.
    slip_std : float | list[float] | None, default 5.0
        Prior std dev for half-logit-scale slipping probability.
    pi_know_mu : float | list[float] | None, default -2.0
        Prior mean for logit-scale initial knowledge probability.
    pi_know_std : float | list[float] | None, default 5.0
        Prior std dev for logit-scale initial knowledge probability.
    pi_b0_know_mu : float | None, default 0.0
        Prior mean for the intercept (b0) in the regression of logit initial knowledge.
        Used only with `InitKnowledgeStrategy.JOINT`.
    pi_b0_know_std : float | None, default 5.0
        Prior std dev for the intercept (b0). Used only with `InitKnowledgeStrategy.JOINT`.
    pi_b1_know_mu : float | list[float] | None, default 0.0
        Prior mean for the covariate slopes (b1). A scalar applies to every covariate, a list
        gives one value per covariate. Used only with `InitKnowledgeStrategy.JOINT`.
    pi_b1_know_std : float | list[float] | None, default 5.0
        Prior std dev for the covariate slopes (b1), scalar or one value per covariate.
        Used only with `InitKnowledgeStrategy.JOINT`.
    pi_sigma_lambda : float | None, default 5.0
        Rate of the Exponential prior on the regression residual std dev (sigma), prior mean 0.2.
        Kept tight because sigma and the b1 slopes are weakly identified, a loose prior inflates
        both. Used only with `InitKnowledgeStrategy.JOINT`.
    post_b0_mu : float | None, default 0.0
        Prior mean for the intercept of the post-test outcome regression.
        Used only when `post_outcome_regression=True`.
    post_b0_std : float | None, default 5.0
        Prior std dev for the post-test outcome intercept.
    post_b_mastery_mu : float | None, default 0.0
        Prior mean for the effect of final mastery on the post-test outcome.
    post_b_mastery_std : float | None, default 5.0
        Prior std dev for the effect of final mastery on the post-test outcome.
    post_b1_mu : float | list[float] | None, default 0.0
        Prior mean for the post-test covariate slopes, scalar or one value per covariate.
    post_b1_std : float | list[float] | None, default 5.0
        Prior std dev for the post-test covariate slopes, scalar or one value per covariate.
    post_sigma_lambda : float | None, default 0.5
        Rate of the Exponential prior on the post-test outcome residual std dev.
    use_defaults : bool, default True
        Whether to fill in missing prior values with defaults (True) or None (False).
        If True, any parameter not explicitly set will be filled with a default value.
        If False, any parameter not explicitly set will be set to None, indicating a non-informative uniform prior.

    """

    # Note: any changes here should also be reflected in the default priors returned by _get_default_priors()
    learn_mu: float | list[float] | None | _UnsetType = _UNSET
    learn_std: float | list[float] | None | _UnsetType = _UNSET
    forget_mu: float | list[float] | None | _UnsetType = _UNSET
    forget_std: float | list[float] | None | _UnsetType = _UNSET
    guess_mu: float | list[float] | None | _UnsetType = _UNSET
    guess_std: float | list[float] | None | _UnsetType = _UNSET
    slip_mu: float | list[float] | None | _UnsetType = _UNSET
    slip_std: float | list[float] | None | _UnsetType = _UNSET

    pi_know_mu: float | list[float] | None | _UnsetType = _UNSET
    pi_know_std: float | list[float] | None | _UnsetType = _UNSET

    def expected_class() -> type[BKTModelBase]:
        """Return the expected BKT model class type for these priors."""
        from stanbkt.models.core.multi import MultiBKT

        return MultiBKT

    def __post_init__(self) -> None:
        """Post-initialization processing to handle default values."""
        if self.use_defaults:
            # unfiltered scalar defaults (covers both CORRECTNESS_ONLY and JOINT keys) -
            # filtering by estimation strategy happens later, at to_dict() call time.
            defaults = MultiPriors._default_scalar_priors()
            for key, default_value in defaults.items():
                if getattr(self, key) is _UNSET:
                    setattr(self, key, default_value)
        else:
            for field in fields(self):
                if getattr(self, field.name) is _UNSET:
                    setattr(self, field.name, None)

    @staticmethod
    def _validate_single(
        single_priors: PriorsBase,
        model_class: type[BKTModelBase],
        estimation_strategy: InitKnowledgeStrategy,
        kc_id: str | None = None,
        n_groups: int | None = None,  # None skips the list length check (groups not known yet)
    ) -> None:
        """Validate that the prior values are compatible with expected types and model structure.

        Checks that each prior value is either a float, a list of floats/None, or None.
        For grouped models, checks that list lengths match the expected number of groups.
        """

        for key, value in single_priors.to_dict(estimation_strategy).items():
            # check type for model class type
            if not issubclass(model_class, MultiPriors.expected_class()):
                raise TypeError(
                    f"Invalid model class: {MultiPriors.__name__} should be used with {MultiPriors.expected_class().__name__}, "
                    f"got {model_class.__name__}"
                )
            if not isinstance(value, (float, list, type(None))):
                raise ValueError(
                    f"Invalid prior value type for {key}: expected float, list, or None, got {type(value).__name__}"
                )
            # if value is a list, check that its length matches n_groups, except for
            # pi_b1_know_mu/std and post_b1_mu/std which are per-covariate lists, not per-group
            if n_groups is not None and isinstance(value, list) and key not in (
                "pi_b1_know_mu", "pi_b1_know_std", "post_b1_mu", "post_b1_std",
            ):
                if len(value) != n_groups:
                    msg = f"Bayesian Prior {key} for KC '{kc_id}' must be a list of length {n_groups}, "
                    msg += f"got list of length {len(value)}"
                    raise ValueError(msg)
            # check std, lambda is positive if not None
            if key.endswith((STD, LAMBDA)) and value is not None:
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
    def _expand_grouped_priors(
        scalar_priors: Union[dict[str, float | None], dict[str, list[float | None]]],
        n_groups: int,
    ) -> dict[str, float | list[float | None] | None]:
        """Expand scalar priors to grouped model format.

        Replicates each scalar prior value ``n_groups`` times to create
        parameter lists suitable for grouped BKT models.
        If a prior value is already a list, it is left unchanged.
        JOINT strategy keys (e.g. pi_b0_know_mu) are left as scalars since
        they are not group-specific.

        Parameters
        ----------
        scalar_priors : dict[str, float | None]
            Mapping of prior names to scalar values.
        n_groups : int
            Number of groups (replicas) for each prior value.

        Returns
        -------
        dict[str, float | list[float | None] | None]
            Mapping of prior names to lists of replicated prior values,
            or scalars for JOINT strategy keys.
        """
        return {
            prior: (
                value
                if prior in JOINT_STRATEGY_KEYS  # leave as is if priors is for joint strategy
                or prior in POST_OUTCOME_STRATEGY_KEYS  # or for the post-outcome regression
                or isinstance(value, list)  # or is already a list
                or n_groups < 1  # or no groups specified
                else ([value] * n_groups)
            )
            for prior, value in scalar_priors.items()
        }

    @staticmethod
    def get_default_priors(
        estimation_type: InitKnowledgeStrategy,
        return_none: bool = False,
        n_groups: int = 0,
    ) -> Union[
        dict[str, float | None],
        dict[str, list[float | None]],
    ]:
        """Return default priors dictionary used for MultiBKT parameters.

        Parameters
        ----------
        estimation_type : InitKnowledgeStrategy
            The strategy used for estimating initial knowledge, which determines which priors are relevant.
        return_none : bool, default False
            If True, return a dict with all keys set to None (improper and non-informative).
            If False, return a dict with default scalar values.
        n_groups : int, default 0
            The number of groups used for the learn, forget, guess, and slip parameters.

        Notes
        -----
        BKT specific priors are modeled as a Normal distributions with means and standard
        deviations specified on the logit scale for learn and forget probability parameters.
        Guess and slip are modeled on the half-logit scale to constrain them to [0, 0.5] on the probability scale.

        In cases where estimation_type = :attr:`stanbkt.models.model_types.InitKnowledgeStrategy.JOINT`, the b0 and b1 parameters
        are modeled as Normal distributions. Additionally, the
        regression residuals is modeled as a positive-valued parameter (pi_sigma_lambda) with a Exponential prior.


        A non-centered parameterization is used:

            logit_pi_know ~ pi_b0_know + pi_b1_know * pretest + pi_sigma * logit_pi_know_z

            Where,
                - pi_b0_know ~ Normal(pi_b0_know_mu, pi_b0_know_std)
                - pi_b1_know ~ Normal(pi_b1_know_mu, pi_b1_know_std)
                - pi_sigma ~ Exponential(pi_sigma_lambda)
                - logit_pi_know_z ~ N(0, 1).

        This model is mathematically equivalent to:

        logit_pi_know ~ Normal(pi_b0_know + pi_b1_know * pretest, pi_sigma)

        but the former (non-centered) parameterization is computationally more efficient and has better
        convergence properties in Stan.
        """

        if estimation_type not in [e for e in InitKnowledgeStrategy]:
            raise ValueError(f"Unsupported prior estimation type: {estimation_type}")

        priors = MultiPriors._expand_grouped_priors(
            MultiPriors._default_scalar_priors(return_none=return_none),
            n_groups=n_groups,
        )
        # need to ignore invalid-argument-type as ty cannot infer that the dict
        # output of _expand_grouped_priors is compatible with the MultiPriors
        # constructor due to the dynamic typing of the values (float | list[float | None] | None)
        return MultiPriors(
            **priors, use_defaults=False  # ty:ignore[invalid-argument-type]
        ).to_dict(estimation_type=estimation_type)


@dataclass
class HierarchicalPriors(PriorsBase):
    """Bayesian priors for hierarchical estimation of BKT parameters.

    Each BKT parameter is modeled as a linear regression with a group-level mean (beta0) and a group-level deviation
    that captures variability across groups. This allows for partial pooling of parameter estimates
    across groups, which can improve estimation for groups with limited data.
    Additionally, it serves as a regularization mechanism for grouped models, preventing overfitting by shrinking
    group-level estimates towards the overall (grand) mean.
    """
