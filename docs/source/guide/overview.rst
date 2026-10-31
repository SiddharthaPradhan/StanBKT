BKT Model Overview
==================

Bayesian Knowledge Tracing (BKT) models a student's mastery of a knowledge component (KC) as a hidden binary
state that is observed through correct and incorrect responses. StanBKT fits the model in Stan, so every
parameter has a full posterior distribution rather than a single estimate.

Parameters
----------

Each KC has five parameters. All of them are estimated on the logit scale and transformed to probabilities.

.. list-table::
   :header-rows: 1
   :widths: 15 55 30

   * - Parameter
     - Meaning
     - Support
   * - ``pi_know``
     - Probability that the student knows the skill before the first interaction.
     - (0, 1)
   * - ``learn``
     - Probability of moving from not knowing to knowing after an interaction.
     - (0, 1)
   * - ``forget``
     - Probability of moving from knowing to not knowing after an interaction.
     - (0, 1)
   * - ``guess``
     - Probability of a correct response without knowing the skill.
     - (0, 0.5) by default
   * - ``slip``
     - Probability of an incorrect response despite knowing the skill.
     - (0, 0.5) by default

``guess`` and ``slip`` are capped at 0.5 (``0.5 * inv_logit(x)``) so that a correct response is always more
likely when the skill is known. This removes the label switching where "known" and "not known" swap meaning.
The cap is set with the ``guess_slip_scale`` constructor argument, which can be any value in (0, 1]. Values above
0.5 no longer rule out label switching, and 1.0 removes the cap.

In :class:`~stanbkt.models.MultiBKT` every parameter is estimated per group. ``pi_know`` can also be estimated
per student, see :doc:`initial_knowledge`.

Hidden Markov formulation
-------------------------

For a student with responses :math:`y_1, \dots, y_T` and hidden states :math:`K_1, \dots, K_T`:

.. math::

   P(K_1 = 1) &= \pi_{know} \\
   P(y_t = 1 \mid K_t = 1) &= 1 - slip, \quad P(y_t = 1 \mid K_t = 0) = guess \\
   P(K_{t+1} = 1 \mid K_t) &= \begin{cases} 1 - forget & K_t = 1 \\ learn & K_t = 0 \end{cases}

The likelihood marginalizes over the hidden states with the forward algorithm. Each KC is an independent
model with its own Stan run, so fitting ten KCs means ten fits.

Predicted quantities
--------------------

Predictions return two values per interaction:

- ``pKnow``: the probability that the student knows the skill at that interaction.
- ``pCorrectness``: the probability of a correct response at that interaction,
  ``pKnow * (1 - slip) + (1 - pKnow) * guess``.

There are two kinds of hidden state:

- **Filtered (online)**, from ``predict`` and ``predict_posterior_*``: ``pKnow`` at interaction :math:`t` is
  conditioned on responses :math:`1, \dots, t-1` only. This is what a tutor would know in real time, and
  ``pCorrectness`` is a genuine prediction of the next response.
- **Smoothed (offline)**, from ``predict_smoothed`` and ``predict_smoothed_posterior_*``: ``pKnow`` is
  conditioned on the full sequence (forward-backward). Use it to describe mastery after the fact, not to
  evaluate predictive accuracy, since it has seen the response it is predicting.

See :doc:`prediction` for the point estimate and full posterior prediction paths.
