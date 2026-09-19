functions{
  /**
  Compute the log density for a slice of students with a scaled forward recursion.
  The recursion runs over time steps with vector operations across students, so the number of
  autodiff nodes depends on the sequence length and not on the number of students.
  Students are sorted by decreasing sequence length, so the active students at each step are a prefix.
  Emission tables: P(correct | not known) = guess, P(correct | known) = 1 - slip.
  */
  real partial_sum(array[] int lengths_slice, // sliced sequence lengths, decreasing (nStudentsInSlice)
                   int start, int end, // slice indexes
                   matrix correct_by_time, // 1 if correct, 0 otherwise (nProblems, nStudents)
                   vector learn_student, // per student parameters, in sorted student order (nStudents)
                   vector forget_student,
                   vector guess_student,
                   vector slip_student,
                   vector pi_know_student // initial probability of knowing (nStudents)
                   ) {
    int n = end - start + 1;
    vector[n] learn_s = learn_student[start:end];
    vector[n] slip_s = slip_student[start:end];
    // emission probabilities are P(y | not known) = not_guess + y * guess_gain and P(y | known) = slip_s + y * slip_gain
    vector[n] not_guess = 1 - guess_student[start:end];
    vector[n] guess_gain = 2 * guess_student[start:end] - 1;
    vector[n] slip_gain = 1 - 2 * slip_s;
    // P(known at next step) = learn + P(known | y) * (1 - forget - learn)
    vector[n] transition_gain = 1 - forget_student[start:end] - learn_s;
    // probability of knowing the skill before observing the current interaction
    vector[n] known = pi_know_student[start:end];

    real target_ = 0.0; // accumulator for the log density
    int active = n; // students that still have interactions at the current step
    for (t in 1:lengths_slice[1]) {
        while (lengths_slice[active] < t) {
            active -= 1;
        }
        vector[active] y = to_vector(correct_by_time[t, start:(start + active - 1)]);
        vector[active] known_now = head(known, active);
        vector[active] joint_known = known_now .* (head(slip_s, active) + y .* head(slip_gain, active));
        vector[active] norm_const = joint_known
            + (1 - known_now) .* (head(not_guess, active) + y .* head(guess_gain, active));
        target_ += sum(log(norm_const));
        // transition to the next interaction, inactive students keep stale values that are never read
        known[1:active] = head(learn_s, active) + (joint_known ./ norm_const) .* head(transition_gain, active);
    }
    return target_;
  }
  /**
  Helper function to apply priors for the parameters. If unif_prior is 0, applies the priors on the logit scale.
  Else stan will automatically apply uniform priors over the parameter space/support.
  */
  void handle_normal_priors_lp(int unif_prior,         // binary indicator for uniform prior
                        row_vector logit_param, // parameter on logit scale
                        array[] real prior_mu,  // prior means for normal distribution
                        array[]  real prior_std // prior stds for normal distribution
                    ){
    if (unif_prior != 1) {
        // Apply normal priors on logit scale
        logit_param ~ normal(prior_mu, prior_std);
    }
    // else stan will automatically apply uniform over the parameter space/support.
  }
  /**
  Initial knowledge probability of one student when generating quantities.
  Students found in the fit (train_idx > 0) reuse their fitted latent value. Under JOINT the rest
  get the regression mean with no noise. Without JOINT every student is in the fit.
  */
  real gq_pi_know(int train_idx, int group_idx, int individual_pi_know, int joint_pi_know,
                  row_vector pi_know,
                  vector b0, row_vector x, vector b1, vector sigma, row_vector z) {
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
    int<lower=1> nProblems;    // number of problems
    int<lower=1> nStudents;    // number of students
    int<lower=1> nGroups;      // number of groups
    array[nStudents] int<lower=1, upper=nGroups> groups; // group assignment for each student (1-based indexing)
    // Note on correctness matrix: -1 = NA, 0 = incorrect, 1 = correct.
    array[nStudents, nProblems] int<lower=-1, upper=1> correctness; // correctness matrix
    array[nStudents] int<lower=1> interaction_lengths; // lengths of each student's interaction sequence

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
    int<lower=0, upper=1> unif_prior_pi_know;
    int<lower=0, upper=1> unif_prior_learn;
    int<lower=0, upper=1> unif_prior_forget;
    int<lower=0, upper=1> unif_prior_guess;
    int<lower=0, upper=1> unif_prior_slip;
    
    int<lower=0, upper=1> individual_pi_know;;

    // JOINT init-knowledge regression, meaningful only when individual_pi_know == 1
    int<lower=0, upper=1> joint_pi_know;      // 1 == InitKnowledgeStrategy.JOINT
    int<lower=1> nTrainStudents;              // fit-time student count backing the pi_know parameters.
                                               // == nStudents at fit time; only differs when generating quantities.
    int<lower=0> nCovariates;                 // number of JOINT covariates (0 when joint_pi_know == 0)
    matrix[nStudents, nCovariates] covariates; // student covariates, in the same order as the students
    // 1-based index of each student in the fit-time population, 0 if not in the fit (only used when generating quantities)
    array[nStudents] int<lower=0, upper=nTrainStudents> train_student_idx;

    real prior_pi_b0_know_mu;
    real<lower=0> prior_pi_b0_know_std;
    array[nCovariates] real prior_pi_b1_know_mu;
    array[nCovariates] real<lower=0> prior_pi_b1_know_std;
    real<lower=0> prior_pi_sigma_lambda;
    int<lower=0, upper=1> unif_prior_pi_b0_know;
    int<lower=0, upper=1> unif_prior_pi_b1_know;
    int<lower=0, upper=1> unif_prior_pi_sigma;
}

transformed data {
    // students sorted by decreasing sequence length, the likelihood does not depend on the student order
    array[nStudents] int student_order = sort_indices_desc(interaction_lengths);
    array[nStudents] int lengths_sorted;
    array[nStudents] int groups_sorted;
    matrix[nProblems, nStudents] correct_by_time = rep_matrix(0, nProblems, nStudents);
    for (sorted_idx in 1:nStudents) {
        int student_idx = student_order[sorted_idx];
        lengths_sorted[sorted_idx] = interaction_lengths[student_idx];
        groups_sorted[sorted_idx] = groups[student_idx];
        for (t in 1:interaction_lengths[student_idx]) {
            correct_by_time[t, sorted_idx] = correctness[student_idx, t];
        }
    }
}

parameters {
    // free logit_pi_know, used only for the non-JOINT path; 0-size when joint_pi_know == 1
    row_vector[joint_pi_know == 1 ? 0 : (individual_pi_know == 1 ? nTrainStudents : nGroups)] logit_pi_know_group;
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
}

transformed parameters {
    // group parameters in probability scale
    // under JOINT the per-student initial knowledge is not stored, it is rebuilt from the regression
    row_vector[joint_pi_know == 1 ? 0 : (individual_pi_know == 1 ? nTrainStudents : nGroups)] pi_know = inv_logit(logit_pi_know_group);
    row_vector[nGroups] learn = inv_logit(logit_learn_group);
    row_vector[nGroups] forget = inv_logit(logit_forget_group);
    // Constrain guess and slip to be <= 0.5
    row_vector[nGroups] guess  = 0.5 * inv_logit(logit_guess_group);
    row_vector[nGroups] slip  = 0.5 * inv_logit(logit_slip_group);
}

model {
    // Bayesian Priors
    if (joint_pi_know == 1) {
        logit_pi_know_z ~ std_normal();
        if (unif_prior_pi_b0_know != 1) {
            pi_b0_know_param ~ normal(prior_pi_b0_know_mu, prior_pi_b0_know_std);
        }
        if (unif_prior_pi_b1_know != 1 && nCovariates > 0) {
            pi_b1_know_param ~ normal(prior_pi_b1_know_mu, prior_pi_b1_know_std);
        }
        if (unif_prior_pi_sigma != 1) {
            pi_sigma_param ~ exponential(prior_pi_sigma_lambda);
        }
    } else if (individual_pi_know == 1) {
        if (unif_prior_pi_know != 1) {
            for (student_idx in 1:nTrainStudents) {
                logit_pi_know_group[student_idx] ~ normal(prior_pi_know_mu[groups[student_idx]], prior_pi_know_std[groups[student_idx]]);
            }
        }
    } else {
        handle_normal_priors_lp(unif_prior_pi_know, logit_pi_know_group, prior_pi_know_mu, prior_pi_know_std);
    }
    handle_normal_priors_lp(unif_prior_learn, logit_learn_group, prior_learn_mu, prior_learn_std);
    handle_normal_priors_lp(unif_prior_forget, logit_forget_group, prior_forget_mu, prior_forget_std);
    handle_normal_priors_lp(unif_prior_guess, logit_guess_group, prior_guess_mu, prior_guess_std);
    handle_normal_priors_lp(unif_prior_slip, logit_slip_group, prior_slip_mu, prior_slip_std);


    // The following variables are based on the parameters and converted into suitable vector or matrix form.
    // While this could have been technically been done in the transformed parameters block,
    // it causes severe memory overhead, both while sampling and saving the fitted model.
    // Declaring the parameters here throws away these intermediate values after using them for estimation.

    // initial probability of knowing the skill for each student, in sorted student order
    vector[nStudents] pi_know_student;
    if (joint_pi_know == 1) {
        // regression on the student covariates with non-centered residual
        vector[nTrainStudents] logit_pi_know_vec = rep_vector(pi_b0_know_param[1], nTrainStudents)
            + pi_sigma_param[1] * to_vector(logit_pi_know_z);
        if (nCovariates > 0) {
            logit_pi_know_vec += covariates * pi_b1_know_param;
        }
        pi_know_student = inv_logit(logit_pi_know_vec[student_order]);
    } else if (individual_pi_know == 1) {
        pi_know_student = to_vector(pi_know[student_order]);
    } else {
        pi_know_student = to_vector(pi_know[groups_sorted]);
    }

    // Parallelized likelihood computation
    int grainsize = 1; // use internal scheduler
    target += reduce_sum(partial_sum, lengths_sorted,
                        grainsize,
                        correct_by_time,
                        to_vector(learn[groups_sorted]), to_vector(forget[groups_sorted]),
                        to_vector(guess[groups_sorted]), to_vector(slip[groups_sorted]),
                        pi_know_student);

}
