Data Format
===========

StanBKT takes interaction data in long format, one row per response.

Columns
-------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - ``ColumnNames`` member
     - Default name
     - Content
   * - ``STUDENT_ID``
     - ``student_id``
     - Student identifier. Converted to string.
   * - ``PROBLEM_ID``
     - ``problem_id``
     - Problem identifier, carried through to predictions.
   * - ``CORRECTNESS``
     - ``correct``
     - Binary outcome, 0 or 1.
   * - ``ORDER``
     - ``timestamp``
     - Any sortable value (datetime, integer, float) giving the order of a student's interactions.
   * - ``KC_ID``
     - ``kc_id``
     - Knowledge component. Optional, all rows belong to a single KC named ``default_kc`` if absent.
   * - ``GROUP``
     - ``group_id``
     - Student group, required by :class:`~stanbkt.models.MultiBKT` only.

Mapping column names
--------------------

If your columns use other names, pass ``column_mapping`` to ``fit``, every ``predict*`` method, ``summary`` and the
plotting functions. Only the columns that differ need an entry.

.. code-block:: python

   from stanbkt.utils import ColumnNames

   col_mapping = {
       ColumnNames.STUDENT_ID: "user_id",
       ColumnNames.CORRECTNESS: "is_correct",
       ColumnNames.ORDER: "order_id",
       ColumnNames.KC_ID: "skill_id",
   }
   model.fit(df, column_mapping=col_mapping)
   model.predict(df, column_mapping=col_mapping)

Validation and cleaning
-----------------------

Before fitting, each KC's data is checked and prepared:

- The student, problem, correctness and order columns must be present, otherwise a ``ValueError`` is raised.
- Correctness must contain only 0 and 1. Missing values are not allowed, so drop unanswered rows first.
- Rows with a missing order value are ignored.
- Two rows with the same order value for the same student in the same KC raise a ``ValueError``.
- Interactions are sorted by order within each student. Students are sorted in natural order
  (``stu_2`` before ``stu_10``), which is also the order used for per student labels in ``summary()``.

How KCs are fitted
------------------

The data is split by KC and each KC is fitted as an independent Stan model. Within a KC, the sequences are
stored as a students by interactions matrix padded with ``-1``, so students can have different numbers of
interactions. A student only enters a KC's fit through the rows they have for that KC.

Because KCs are independent, ``fit`` is additive: calling it again with new KCs adds their fits to the model.
Refitting a KC that already has a fit raises an error unless ``overwrite_kcs=True``.

.. code-block:: python

   model.fit(df[df["kc_id"] == "kc_0"])
   model.fit(df[df["kc_id"] == "kc_1"])  # adds kc_1
   model.fit(df[df["kc_id"] == "kc_0"], overwrite_kcs=True)  # refits kc_0

By default KCs are fitted concurrently, see :doc:`performance`.

See the :doc:`../examples/08_multiple_kcs` notebook for a worked example.
