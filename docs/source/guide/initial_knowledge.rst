Initial Knowledge
=================

StanBKT has three ways to model ``pi_know``, the probability that a student knows the skill before their first
interaction. They are set with two constructor arguments and work the same way in ``StandardBKT`` and ``MultiBKT``.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Mode
     - Arguments
     - ``pi_know``
   * - Population (default)
     - ``individual_initial_knowledge=False``
     - One value per KC (``StandardBKT``) or per group (``MultiBKT``).
   * - Individual
     - ``individual_initial_knowledge=True``, ``init_knowledge_strategy=CORRECTNESS_ONLY``
     - One free value per fitted student, estimated from their responses.
   * - JOINT
     - ``individual_initial_knowledge=True``, ``init_knowledge_strategy=JOINT``
     - Per student, regressed on student covariates.

``init_knowledge_strategy`` must stay ``CORRECTNESS_ONLY`` (the default) unless ``individual_initial_knowledge``
is True, otherwise the constructor raises a ``ValueError``.

Population initial knowledge
----------------------------

All students in a KC (or a group) share ``pi_know``. This is the classic BKT model and the most robust choice
with few interactions per student.

Individual initial knowledge
----------------------------

Each student gets their own ``pi_know``, with the KC (or group) prior from :doc:`priors` as a shared prior.
Students with short sequences carry little information, so their estimates stay close to the prior.

.. code-block:: python

   from stanbkt.models import StandardBKT

   model = StandardBKT(individual_initial_knowledge=True)
   model.fit(df)
   model.summary()  # logit_pi_know_group and pi_know have one row per student

Because each ``pi_know`` belongs to a fitted student, predictions can only be made for those students. Students
in the prediction data that were not part of the KC's fit are dropped with a warning, and a ``ValueError`` is
raised if no fitted students remain. Students are matched by ID through the ``student2index`` map saved with
each fit.

JOINT initial knowledge
-----------------------

JOINT regresses each student's logit ``pi_know`` on student level covariates such as a pretest score, so
information is shared across students and unseen students still get a prediction from their covariates.

.. code-block:: python

   from stanbkt.models import StandardBKT, InitKnowledgeStrategy

   model = StandardBKT(
       individual_initial_knowledge=True,
       init_knowledge_strategy=InitKnowledgeStrategy.JOINT,
   )
   model.fit(df, student_covariates=covariates_df)
   model.predict(new_df, student_covariates=new_covariates_df)

See :doc:`joint_strategy` for the model equation, the covariate frame, priors and how to read the results.

See the :doc:`../examples/06_individual_initial_knowledge` notebook for a worked example.
