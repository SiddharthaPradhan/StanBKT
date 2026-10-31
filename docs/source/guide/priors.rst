Priors
======

Every BKT parameter has a normal prior on its unconstrained scale. Priors are given per KC with
:class:`~stanbkt.models.StandardPriors` for ``StandardBKT`` and :class:`~stanbkt.models.MultiPriors` for ``MultiBKT``.
Passing the wrong class for the model raises a ``TypeError``.

Scale
-----

``pi_know``, ``learn`` and ``forget`` are modeled as ``inv_logit(x)`` and ``guess`` and ``slip`` as
``guess_slip_scale * inv_logit(x)``, where ``guess_slip_scale`` is a model constructor argument that defaults to
0.5 (the half-logit scale). The ``*_mu`` and ``*_std`` fields are the mean and standard deviation of a normal prior on
``x``, not on the probability.

.. code-block:: python

   import numpy as np

   def logit(p):
       return np.log(p / (1 - p))

   1 / (1 + np.exp(2.0))  # 0.12, inv_logit(-2.0), the default prior mean for pi_know
   logit(0.3)  # -0.85, the logit of a learn rate of 0.3
   logit(0.1 / 0.5)  # -1.39, the half-logit of a guess of 0.1 (guess_slip_scale=0.5)
   logit(0.1 / 0.3)  # -0.69, the same guess with guess_slip_scale=0.3

The same ``guess_mu`` or ``slip_mu`` therefore implies a different probability under a different
``guess_slip_scale``. The default of -1.0 is a guess of ``0.5 * inv_logit(-1.0) = 0.13`` with the default scale,
but ``0.27`` with ``guess_slip_scale=1.0``. Rescale the prior means when changing it.

A standard deviation of 5 on the logit scale is very wide. After the transformation most of its mass sits near
0 and 1, so it is weakly informative about the location but not flat on the probability scale.

Defaults
--------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Parameter
     - ``*_mu``
     - ``*_std``
   * - ``learn``
     - 0.0
     - 5.0
   * - ``forget``
     - -2.0
     - 5.0
   * - ``guess`` (half-logit by default)
     - -1.0
     - 5.0
   * - ``slip`` (half-logit by default)
     - -1.0
     - 5.0
   * - ``pi_know``
     - -2.0
     - 5.0

The JOINT regression (:doc:`joint_strategy`) and post-test regression (:doc:`post_outcome`) coefficients default to
normal priors with mean 0 and standard deviation 5. The post-test residual standard deviation has an Exponential
prior with rate 0.5, and the JOINT residual standard deviation one with rate 5 (mean 0.2), kept tight because it is
weakly identified.

Overriding defaults
-------------------

Fields you do not set keep their defaults.

.. code-block:: python

   from stanbkt.models import StandardPriors

   # informative prior: learning rates near inv_logit(-2) = 0.12
   priors = StandardPriors(learn_mu=-2.0, learn_std=0.5)
   model.fit(df, priors=priors)

Uniform priors
--------------

Setting either ``*_mu`` or ``*_std`` of a parameter to ``None`` removes its prior, which gives an improper flat
prior on the logit scale. ``use_defaults=False`` sets every field you do not pass to ``None``, so the following
has no priors at all, and an MLE fit with it is plain maximum likelihood.

.. code-block:: python

   uniform = StandardPriors(use_defaults=False)

   # flat prior on forget only, defaults elsewhere
   flat_forget = StandardPriors(forget_mu=None)

The covariate slopes (``pi_b1_know_*`` and ``post_b1_*``) can also be set per covariate by using ``None`` inside a
list. Here the first covariate has the prior ``Normal(1, 3)`` and the second a flat prior:

.. code-block:: python

   priors = StandardPriors(pi_b1_know_mu=[1.0, None], pi_b1_know_std=[3.0, 3.0])

Flat priors can leave a KC weakly identified with little data. Check R-hat and ESS (:doc:`summaries`) when you use them.

Per KC priors
-------------

A single priors object applies to every KC. To use different priors per KC, pass a dict keyed by KC ID. KCs that
are not in the dict use the default priors.

.. code-block:: python

   priors = {
       "kc_0": StandardPriors(learn_mu=-1.0),
       "kc_1": StandardPriors(use_defaults=False),
   }
   model.fit(df, priors=priors)

Per group priors
----------------

:class:`~stanbkt.models.MultiPriors` takes a scalar, shared by every group, or a list with one value per group for
``learn``, ``forget``, ``guess``, ``slip`` and ``pi_know``. A list must match the number of groups in every KC it is
applied to, in the group order described in :doc:`choosing_a_model`. When KCs have different groups, use a per KC
dict. Setting a parameter's ``*_mu`` or ``*_std`` to ``None`` gives a flat prior for that parameter in every group,
and ``None`` inside a list gives a flat prior in that group only. Here the first group has the prior ``Normal(-2, 1)``
on ``learn`` and the second a flat prior:

.. code-block:: python

   priors = MultiPriors(learn_mu=[-2.0, None], learn_std=[1.0, 1.0])

.. code-block:: python

   from stanbkt.models import MultiPriors

   priors = MultiPriors(learn_mu=[-2.0, -1.0, 0.0], slip_std=1.0)
   model.fit(grouped_df, priors=priors)

With individualized initial knowledge, each student's ``pi_know`` uses the prior of their group.

See the :doc:`../examples/04_priors` notebook for a worked example.
