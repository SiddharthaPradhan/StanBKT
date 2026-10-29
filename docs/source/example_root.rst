Example Gallery
===============

Getting Started
---------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Setup
      :text-align: center
      :class-card: example-card
      :link: ./examples/01_setup.ipynb

      Learn how to set up StanBKT and prepare your environment for Bayesian Knowledge Tracing modeling.

   .. grid-item-card:: Simple Example
      :text-align: center
      :class-card: example-card
      :link: ./examples/02_simple_example.ipynb

      Walk through a basic example to understand the fundamentals of BKT and how to fit models.

Concepts
--------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Priors
      :text-align: center
      :class-card: example-card
      :link: ./examples/04_priors.ipynb

      Compare default, informative and uniform priors, set priors per KC, and use per group priors with MultiPriors.

   .. grid-item-card:: Inference Methods
      :text-align: center
      :class-card: example-card
      :link: ./examples/05_inference_methods.ipynb

      Fit one dataset with MCMC, VB, MLE and Pathfinder, and compare speed, summaries and prediction support.

Advanced Models
---------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Multi-BKT
      :text-align: center
      :class-card: example-card
      :link: ./examples/03_multi_BKT_example.ipynb

      Fit grouped BKT parameters with MultiBKT, where each student group gets its own set of BKT parameters.

   .. grid-item-card:: Individual Initial Knowledge
      :text-align: center
      :class-card: example-card
      :link: ./examples/06_individual_initial_knowledge.ipynb

      Estimate a separate initial knowledge per student and see how unseen students are handled.

   .. grid-item-card:: JOINT Initial Knowledge
      :text-align: center
      :class-card: example-card
      :link: ./examples/07_joint_initial_knowledge.ipynb

      Regress initial knowledge on student covariates, choose a prior for sigma, and predict new students.

Workflows
---------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Multiple KCs
      :text-align: center
      :class-card: example-card
      :link: ./examples/08_multiple_kcs.ipynb

      Map dataset column names, fit many KCs concurrently, add KCs later, and refit with per KC priors.

   .. grid-item-card:: Saving and Large Data
      :text-align: center
      :class-card: example-card
      :link: ./examples/09_save_load_large_data.ipynb

      Save and load models, keep fits on disk with low_memory, and predict on large data with the summary methods.

Model Checking
--------------

.. grid:: 1 2 3 3
   :gutter: 2

   .. grid-item-card:: Diagnostics and Plotting
      :text-align: center
      :class-card: example-card
      :link: ./examples/10_diagnostics_and_plotting.ipynb

      Check R-hat and ESS, plot traces and posteriors, and plot posterior correctness for grouped models.

   .. grid-item-card:: Evaluation
      :text-align: center
      :class-card: example-card
      :link: ./examples/11_evaluation.ipynb

      Evaluate on held out students with accuracy, RMSE and AUC, and see why smoothed predictions overstate accuracy.

.. toctree::
   :hidden:
   :caption: Getting Started

   ./examples/01_setup.ipynb
   ./examples/02_simple_example.ipynb

.. toctree::
   :hidden:
   :caption: Concepts

   ./examples/04_priors.ipynb
   ./examples/05_inference_methods.ipynb

.. toctree::
   :hidden:
   :caption: Advanced Models

   ./examples/03_multi_BKT_example.ipynb
   ./examples/06_individual_initial_knowledge.ipynb
   ./examples/07_joint_initial_knowledge.ipynb

.. toctree::
   :hidden:
   :caption: Workflows

   ./examples/08_multiple_kcs.ipynb
   ./examples/09_save_load_large_data.ipynb

.. toctree::
   :hidden:
   :caption: Model Checking

   ./examples/10_diagnostics_and_plotting.ipynb
   ./examples/11_evaluation.ipynb
