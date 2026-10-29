Choosing a Model
================

StanBKT has two model classes. Both use the same Stan program and share every method and constructor option.

StandardBKT
-----------

:class:`~stanbkt.models.StandardBKT` estimates one set of parameters per KC, shared by all students. Use it as
the default, and whenever you have no reason to believe that groups of students learn differently.

.. code-block:: python

   from stanbkt.models import StandardBKT, StandardPriors

   model = StandardBKT()
   model.fit(df, priors=StandardPriors())

MultiBKT
--------

:class:`~stanbkt.models.MultiBKT` estimates a separate set of parameters per group of students within each KC,
for example per class, condition of an experiment, or tutoring system version. The data needs a group column
(``ColumnNames.GROUP``, ``group_id`` by default), and each student belongs to one group within a KC. Grouping is
by student only, there is no problem level grouping.

.. code-block:: python

   from stanbkt.models import MultiBKT, MultiPriors

   model = MultiBKT()
   model.fit(grouped_df, priors=MultiPriors())
   model.summary().loc["kc_0"]  # one row per group for each parameter

Group alignment at prediction time
----------------------------------

The groups seen in each KC are stored with the fit (``group2index``) and saved with the model. At prediction
time groups are matched by name, so the prediction data can contain a subset of the groups in any order. A
group that was not part of the KC's fit raises a ``ValueError``.

Groups are indexed per KC in order of first appearance, with students in natural sort order. ``summary()``
replaces the Stan indexes with the group names, so you rarely need the index itself. Per group prior lists in
:class:`~stanbkt.models.MultiPriors` follow the same order, see :doc:`priors`.

Which to use
------------

- Start with ``StandardBKT``. It has fewer parameters and is easier to identify with small data.
- Use ``MultiBKT`` when a group variable is part of the question (comparing learning rates between conditions),
  or when groups clearly differ and you have enough students per group.
- Compare the per group posteriors in ``summary()``. If their intervals overlap heavily, the grouping adds little.

Both models support individualized initial knowledge (:doc:`initial_knowledge`) and the post-test outcome
regression (:doc:`post_outcome`).
