Example Gallery
===============

.. grid:: 1 2 2 3
   :margin: 4 4 0 0
   :gutter: 2

   .. grid-item-card:: Getting Started
      :text-align: center
      :link: ./examples/01_setup.ipynb

      Setup Guide

      +++

      Learn how to set up StanBKT and prepare your environment for Bayesian Knowledge Tracing modeling.

   .. grid-item-card:: Simple Example
      :text-align: center
      :link: ./examples/02_simple_example.ipynb

      Quick Start

      +++

      Walk through a basic example to understand the fundamentals of BKT and how to fit models.

   .. grid-item-card:: Multi-BKT Modeling
      :text-align: center
      :link: ./examples/03_multi_BKT_example.ipynb

      Advanced

      +++

      Fit grouped BKT parameters with MultiBKT, where each student group gets its own set of BKT parameters.

   .. grid-item-card:: Priors
      :text-align: center
      :link: ./examples/04_priors.ipynb

      Concepts

      +++

      Compare default, informative and uniform priors, set priors per KC, and use per group priors with MultiPriors.

   .. grid-item-card:: Inference Methods
      :text-align: center
      :link: ./examples/05_inference_methods.ipynb

      Concepts

      +++

      Fit one dataset with MCMC, VB, MLE and Pathfinder, and compare speed, summaries and prediction support.

   .. grid-item-card:: Individual Initial Knowledge
      :text-align: center
      :link: ./examples/06_individual_initial_knowledge.ipynb

      Initial Knowledge

      +++

      Estimate a separate initial knowledge per student and see how unseen students are handled.

   .. grid-item-card:: JOINT Initial Knowledge
      :text-align: center
      :link: ./examples/07_joint_initial_knowledge.ipynb

      Initial Knowledge

      +++

      Regress initial knowledge on student covariates, choose a prior for sigma, and predict new students.

   .. grid-item-card:: Multiple KCs
      :text-align: center
      :link: ./examples/08_multiple_kcs.ipynb

      Workflows

      +++

      Map dataset column names, fit many KCs concurrently, add KCs later, and refit with per KC priors.

   .. grid-item-card:: Saving and Large Data
      :text-align: center
      :link: ./examples/09_save_load_large_data.ipynb

      Workflows

      +++

      Save and load models, keep fits on disk with low_memory, and predict on large data with the summary methods.

   .. grid-item-card:: Diagnostics and Plotting
      :text-align: center
      :link: ./examples/10_diagnostics_and_plotting.ipynb

      Workflows

      +++

      Check R-hat and ESS, plot traces and posteriors, and plot posterior correctness for grouped models.

   .. grid-item-card:: Evaluation
      :text-align: center
      :link: ./examples/11_evaluation.ipynb

      Workflows

      +++

      Evaluate on held out students with accuracy, RMSE and AUC, and see why smoothed predictions overstate accuracy.

.. toctree::
   :hidden:

   ./examples/01_setup.ipynb
   ./examples/02_simple_example.ipynb
   ./examples/03_multi_BKT_example.ipynb
   ./examples/04_priors.ipynb
   ./examples/05_inference_methods.ipynb
   ./examples/06_individual_initial_knowledge.ipynb
   ./examples/07_joint_initial_knowledge.ipynb
   ./examples/08_multiple_kcs.ipynb
   ./examples/09_save_load_large_data.ipynb
   ./examples/10_diagnostics_and_plotting.ipynb
   ./examples/11_evaluation.ipynb
