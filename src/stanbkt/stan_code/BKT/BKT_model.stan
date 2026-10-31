functions {
  #include final_mastery.stan
  
  /**
  Compute the log density for a slice of students with a scaled forward recursion.
  The recursion runs over time steps with vector operations across students, so the number of
  autodiff nodes depends on the sequence length and not on the number of students.
  Students are sorted by decreasing sequence length, so the active students at each step are a prefix.
  Emission tables: P(correct | not known) = guess, P(correct | known) = 1 - slip.
  */
  real partial_sum(array[] real pi_know_slice, // sliced initial probability of knowing, sorted student order
                   int start,
                   int end, // slice indexes
                   array[] int lengths_sorted, // sequence lengths, decreasing (nStudents)
                   array[,] int correct_by_time, // 1 if correct, 0 otherwise (nProblems, nStudents)
                   array[] int groups_sorted, // group of each student, sorted student order (nStudents)
                   row_vector learn, // group parameters (nGroups)
                   row_vector forget,
                   row_vector guess,
                   row_vector slip) {
    int n = end - start + 1;
    array[n] int g = groups_sorted[start : end];
    vector[n] learn_s = to_vector(learn[g]);
    vector[n] slip_s = to_vector(slip[g]);
    vector[n] guess_s = to_vector(guess[g]);
    // emission tables stacked as [P(incorrect | state); P(correct | state)], indexed by student + n * y.
    // indexing with data creates no autodiff nodes, unlike mixing the tables with y arithmetically
    vector[2 * n] emit_known = append_row(slip_s, 1 - slip_s);
    vector[2 * n] emit_unknown = append_row(1 - guess_s, guess_s);
    // P(known at next step) = learn + P(known | y) * (1 - forget - learn)
    vector[n] transition_gain = 1 - to_vector(forget[g]) - learn_s;
    // probability of knowing the skill before observing the current interaction
    vector[n] known = to_vector(pi_know_slice);
    
    real target_ = 0.0; // accumulator for the log density
    int active = n; // students that still have interactions at the current step
    array[n] int idx;
    for (t in 1 : lengths_sorted[start]) {
      while (lengths_sorted[start + active - 1] < t) {
        active -= 1;
      }
      // correct_by_time[t] is contiguous over students
      for (i in 1 : active) {
        idx[i] = i + n * correct_by_time[t, start + i - 1];
      }
      vector[active] known_now = head(known, active);
      vector[active] joint_known = known_now .* emit_known[idx[1 : active]];
      vector[active] norm_const = joint_known
                                  + (1 - known_now)
                                    .* emit_unknown[idx[1 : active]];
      target_ += sum(log(norm_const));
      // transition to the next interaction, inactive students keep stale values that are never read
      known[1 : active] = head(learn_s, active)
                          + (joint_known ./ norm_const)
                            .* head(transition_gain, active);
    }
    return target_;
  }
  /**
  Helper function to apply priors for the parameters. Applies the priors on the logit scale to the groups
  whose unif_prior is 0. Else stan will automatically apply uniform priors over the parameter space/support.
  */
  void handle_normal_priors_lp(array[] int unif_prior, // binary indicator for uniform prior, one per group
                               row_vector logit_param, // parameter on logit scale
                               array[] real prior_mu, // prior means for normal distribution
                               array[] real prior_std // prior stds for normal distribution
                               ) {
    for (group_idx in 1 : num_elements(logit_param)) {
      if (unif_prior[group_idx] != 1) {
        // Apply normal priors on logit scale
        logit_param[group_idx] ~ normal(prior_mu[group_idx], prior_std[group_idx]);
      }
      // else stan will automatically apply uniform over the parameter space/support.
    }
  }
  /**
  Initial knowledge probability of one student when generating quantities.
  Students found in the fit (train_idx > 0) reuse their fitted latent value. Under JOINT the rest
  get the regression mean with no noise. Without JOINT every student is in the fit.
  */
  real gq_pi_know(int train_idx,
                  int group_idx,
                  int individual_pi_know,
                  int joint_pi_know,
                  row_vector pi_know,
                  vector b0,
                  row_vector x,
                  vector b1,
                  vector sigma,
                  row_vector z) {
    if (individual_pi_know == 0) {
      return pi_know[group_idx];
    }
    if (joint_pi_know == 1) {
      real mu = b0[1];
      if (size(x) > 0) {
        mu += x * b1;
      }
      if (train_idx > 0) {
        mu += sigma[1] * z[train_idx];
      }
      return inv_logit(mu);
    }
    return pi_know[train_idx];
  }
}
data {
  int<lower=1> nProblems; // number of problems
  int<lower=1> nStudents; // number of students
  int<lower=1> nGroups; // number of groups
  array[nStudents] int<lower=1, upper=nGroups> groups; // group assignment for each student (1-based indexing)
  // Note on correctness matrix: -1 = NA, 0 = incorrect, 1 = correct.
  array[nStudents, nProblems] int<lower=-1, upper=1> correctness; // correctness matrix
  array[nStudents] int<lower=1> interaction_lengths; // lengths of each student's interaction sequence
  int<lower=1> nObs; // total number of interactions, sum of interaction_lengths
  int<lower=1> grainsize; // reduce_sum slice size, set from python
  real<lower=0, upper=1> guess_slip_scale; // guess and slip are scale * inv_logit(x), so capped at scale
  
  // priors for the parameters
  array[nGroups] real prior_pi_know_mu;
  array[nGroups] real<lower=0> prior_pi_know_std;
  array[nGroups] real prior_learn_mu;
  array[nGroups] real<lower=0> prior_learn_std;
  array[nGroups] real prior_forget_mu;
  array[nGroups] real<lower=0> prior_forget_std;
  array[nGroups] real prior_guess_mu;
  array[nGroups] real<lower=0> prior_guess_std;
  array[nGroups] real prior_slip_mu;
  array[nGroups] real<lower=0> prior_slip_std;
  
  // binary indicator whether to use non-infomative uniform priors.
  array[nGroups] int<lower=0, upper=1> unif_prior_pi_know;
  array[nGroups] int<lower=0, upper=1> unif_prior_learn;
  array[nGroups] int<lower=0, upper=1> unif_prior_forget;
  array[nGroups] int<lower=0, upper=1> unif_prior_guess;
  array[nGroups] int<lower=0, upper=1> unif_prior_slip;
  
  int<lower=0, upper=1> individual_pi_know;
  
  // JOINT init-knowledge regression, meaningful only when individual_pi_know == 1
  int<lower=0, upper=1> joint_pi_know; // 1 == InitKnowledgeStrategy.JOINT
  int<lower=1> nTrainStudents; // fit-time student count backing the pi_know parameters.
  // == nStudents at fit time; only differs when generating quantities.
  int<lower=0> nCovariates; // number of JOINT covariates (0 when joint_pi_know == 0)
  matrix[nStudents, nCovariates] covariates; // student covariates, in the same order as the students
  // 1-based index of each student in the fit-time population, 0 if not in the fit (only used when generating quantities)
  array[nStudents] int<lower=0, upper=nTrainStudents> train_student_idx;
  
  real prior_pi_b0_know_mu;
  real<lower=0> prior_pi_b0_know_std;
  array[nCovariates] real prior_pi_b1_know_mu;
  array[nCovariates] real<lower=0> prior_pi_b1_know_std;
  real<lower=0> prior_pi_sigma_lambda;
  int<lower=0, upper=1> unif_prior_pi_b0_know;
  array[nCovariates] int<lower=0, upper=1> unif_prior_pi_b1_know;
  int<lower=0, upper=1> unif_prior_pi_sigma;
  
  // post-test outcome regression on final mastery, independent of the init-knowledge strategy above
  int<lower=0, upper=1> post_outcome_regression; // 1 == post-test outcome regression enabled
  vector[post_outcome_regression == 1 ? nStudents : 0] post_outcome; // post-test outcome, original student order
  int<lower=0> nPostCovariates; // number of post-test covariates (0 when post_outcome_regression == 0)
  matrix[nStudents, nPostCovariates] post_covariates; // post-test covariates, original student order
  
  real prior_post_b0_mu;
  real<lower=0> prior_post_b0_std;
  real prior_post_b_mastery_mu;
  real<lower=0> prior_post_b_mastery_std;
  array[nPostCovariates] real prior_post_b1_mu;
  array[nPostCovariates] real<lower=0> prior_post_b1_std;
  real<lower=0> prior_post_sigma_lambda;
  int<lower=0, upper=1> unif_prior_post_b0;
  int<lower=0, upper=1> unif_prior_post_b_mastery;
  array[nPostCovariates] int<lower=0, upper=1> unif_prior_post_b1;
}
transformed data {
  // students sorted by decreasing sequence length, the likelihood does not depend on the student order
  array[nStudents] int student_order = sort_indices_desc(interaction_lengths);
  array[nStudents] int lengths_sorted;
  array[nStudents] int groups_sorted;
  // time major so each time step reads a contiguous run of students
  array[nProblems, nStudents] int correct_by_time = rep_array(0, nProblems,
                                                              nStudents);
  if (nObs != sum(interaction_lengths)) {
    reject("nObs must equal sum(interaction_lengths), found ", nObs, " and ",
           sum(interaction_lengths));
  }
  for (sorted_idx in 1 : nStudents) {
    int student_idx = student_order[sorted_idx];
    lengths_sorted[sorted_idx] = interaction_lengths[student_idx];
    groups_sorted[sorted_idx] = groups[student_idx];
    for (t in 1 : interaction_lengths[student_idx]) {
      correct_by_time[t, sorted_idx] = correctness[student_idx, t];
    }
  }
}
parameters {
  // free logit_pi_know, used only for the non-JOINT path; 0-size when joint_pi_know == 1
  row_vector[joint_pi_know == 1 ? 0
             : (individual_pi_know == 1 ? nTrainStudents : nGroups)] logit_pi_know_group;
  // group-level parameters (logit scale)
  row_vector[nGroups] logit_learn_group;
  row_vector[nGroups] logit_forget_group;
  row_vector[nGroups] logit_guess_group;
  row_vector[nGroups] logit_slip_group;
  
  // JOINT init-knowledge regression parameters, 0-size when joint_pi_know == 0
  vector[joint_pi_know] pi_b0_know_param;
  vector[joint_pi_know == 1 ? nCovariates : 0] pi_b1_know_param;
  vector<lower=0>[joint_pi_know] pi_sigma_param;
  row_vector[joint_pi_know == 1 ? nTrainStudents : 0] logit_pi_know_z;
  
  // post-test outcome regression parameters, 0-size when post_outcome_regression == 0
  vector[post_outcome_regression] post_b0_param;
  vector[post_outcome_regression] post_b_mastery_param;
  vector[post_outcome_regression == 1 ? nPostCovariates : 0] post_b1_param;
  vector<lower=0>[post_outcome_regression] post_sigma_param;
}
transformed parameters {
  // group parameters in probability scale
  // under JOINT the per-student initial knowledge is not stored, it is rebuilt from the regression
  row_vector[joint_pi_know == 1 ? 0
             : (individual_pi_know == 1 ? nTrainStudents : nGroups)] pi_know = inv_logit(
                                                                    logit_pi_know_group);
  row_vector[nGroups] learn = inv_logit(logit_learn_group);
  row_vector[nGroups] forget = inv_logit(logit_forget_group);
  // Constrain guess and slip to be <= guess_slip_scale (0.5 by default)
  row_vector[nGroups] guess = guess_slip_scale * inv_logit(logit_guess_group);
  row_vector[nGroups] slip = guess_slip_scale * inv_logit(logit_slip_group);
}
model {
  // Bayesian Priors
  if (joint_pi_know == 1) {
    logit_pi_know_z ~ std_normal();
    if (unif_prior_pi_b0_know != 1) {
      pi_b0_know_param ~ normal(prior_pi_b0_know_mu, prior_pi_b0_know_std);
    }
    for (i in 1:nCovariates) {
      if (unif_prior_pi_b1_know[i] != 1) {
        pi_b1_know_param[i] ~ normal(prior_pi_b1_know_mu[i], prior_pi_b1_know_std[i]);
      }
    }
    if (unif_prior_pi_sigma != 1) {
      pi_sigma_param ~ exponential(prior_pi_sigma_lambda);
    }
  } else if (individual_pi_know == 1) {
    for (student_idx in 1 : nTrainStudents) {
      if (unif_prior_pi_know[groups[student_idx]] != 1) {
        logit_pi_know_group[student_idx] ~ normal(
                                                  prior_pi_know_mu[groups[student_idx]],
                                                  prior_pi_know_std[groups[student_idx]]);
      }
    }
  } else {
    handle_normal_priors_lp(unif_prior_pi_know, logit_pi_know_group,
      prior_pi_know_mu, prior_pi_know_std);
  }
  handle_normal_priors_lp(unif_prior_learn, logit_learn_group,
    prior_learn_mu, prior_learn_std);
  handle_normal_priors_lp(unif_prior_forget, logit_forget_group,
    prior_forget_mu, prior_forget_std);
  handle_normal_priors_lp(unif_prior_guess, logit_guess_group,
    prior_guess_mu, prior_guess_std);
  handle_normal_priors_lp(unif_prior_slip, logit_slip_group, prior_slip_mu,
    prior_slip_std);
  
  if (post_outcome_regression == 1) {
    if (unif_prior_post_b0 != 1) {
      post_b0_param ~ normal(prior_post_b0_mu, prior_post_b0_std);
    }
    if (unif_prior_post_b_mastery != 1) {
      post_b_mastery_param ~ normal(prior_post_b_mastery_mu,
                                    prior_post_b_mastery_std);
    }
    for (i in 1:nPostCovariates) {
      if (unif_prior_post_b1[i] != 1) {
        post_b1_param[i] ~ normal(prior_post_b1_mu[i], prior_post_b1_std[i]);
      }
    }
    post_sigma_param ~ exponential(prior_post_sigma_lambda);
  }
  
  // The following variables are based on the parameters and converted into suitable vector or matrix form.
  // While this could have been technically been done in the transformed parameters block,
  // it causes severe memory overhead, both while sampling and saving the fitted model.
  // Declaring the parameters here throws away these intermediate values after using them for estimation.
  
  // initial probability of knowing the skill for each student, in sorted student order
  vector[nStudents] pi_know_student;
  if (joint_pi_know == 1) {
    // regression on the student covariates with non-centered residual
    vector[nTrainStudents] logit_pi_know_vec = rep_vector(
                                                          pi_b0_know_param[1],
                                                          nTrainStudents)
                                               + pi_sigma_param[1]
                                                 * to_vector(logit_pi_know_z);
    if (nCovariates > 0) {
      logit_pi_know_vec += covariates * pi_b1_know_param;
    }
    pi_know_student = inv_logit(logit_pi_know_vec[student_order]);
  } else if (individual_pi_know == 1) {
    pi_know_student = to_vector(pi_know[student_order]);
  } else {
    pi_know_student = to_vector(pi_know[groups_sorted]);
  }
  
  // Parallelized likelihood computation, grainsize is chosen in python from the number of students and threads
  target += reduce_sum(partial_sum, to_array_1d(pi_know_student), grainsize,
                       lengths_sorted, correct_by_time, groups_sorted, learn,
                       forget, guess, slip);
  
  // post-test outcome regression on final (filtered) mastery, jointly estimated with the BKT fit above.
  // Runs a second forward pass in original (unsorted) student order since it can't ride on the
  // reduce_sum call above: reduce_sum only ever combines its shards into a single summed real, so a
  // per-student vector of terminal states can't be extracted from it.
  if (post_outcome_regression == 1) {
    vector[nStudents] pi_know_student_orig;
    if (joint_pi_know == 1) {
      vector[nStudents] logit_pi_know_vec = rep_vector(pi_b0_know_param[1],
                                                       nStudents)
                                            + pi_sigma_param[1]
                                              * to_vector(logit_pi_know_z);
      if (nCovariates > 0) {
        logit_pi_know_vec += covariates * pi_b1_know_param;
      }
      pi_know_student_orig = inv_logit(logit_pi_know_vec);
    } else if (individual_pi_know == 1) {
      pi_know_student_orig = to_vector(pi_know);
    } else {
      pi_know_student_orig = to_vector(pi_know[groups]);
    }
    
    vector[nStudents] final_pknow = final_pknow_forward(interaction_lengths,
                                      correctness, to_vector(learn[groups]),
                                      to_vector(forget[groups]),
                                      to_vector(guess[groups]),
                                      to_vector(slip[groups]),
                                      pi_know_student_orig);
    
    vector[nStudents] post_mu = rep_vector(post_b0_param[1], nStudents)
                                + post_b_mastery_param[1] * final_pknow;
    if (nPostCovariates > 0) {
      post_mu += post_covariates * post_b1_param;
    }
    post_outcome ~ normal(post_mu, post_sigma_param[1]);
  }
}
