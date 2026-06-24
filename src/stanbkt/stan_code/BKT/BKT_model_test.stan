functions{
  /**
  Compute the log density for a slice of students.
  Each student's row is treated as an ordered interaction sequence.
  Transitions and emissions may vary by both student and problem groups.
  */
  real partial_sum(array[,] int correctness,
                   int start, int end,
                   array[,] int problem_sequence,
                   array[] int interaction_lengths,
                   matrix learn, matrix forget, matrix guess, matrix slip,
                   vector pi_know,
                   array[] int studentGroupsInit,
                   array[] int studentGroupsTransition,
                   array[] int studentGroupsEmission,
                   array[] int problemGroupsTransition,
                   array[] int problemGroupsEmission) {

    real target_ = 0.0;
    int localStudentIdx = 1;

    for (studentIdx in start:end) {
      int L = interaction_lengths[studentIdx];
      int init_group = studentGroupsInit[studentIdx];
      int studentTransitionGroup = studentGroupsTransition[studentIdx];
      int studentEmissionGroup = studentGroupsEmission[studentIdx];
      vector[2] latent_mastery = to_vector([1 - pi_know[init_group], pi_know[init_group]]);

      for (t in 1:L) {
        int problemIdx = problem_sequence[localStudentIdx, t];
        int problemTransitionGroup = problemGroupsTransition[problemIdx];
        int problemEmissionGroup = problemGroupsEmission[problemIdx];
        real learn_t = learn[studentTransitionGroup, problemTransitionGroup];
        real forget_t = forget[studentTransitionGroup, problemTransitionGroup];
        real guess_t = guess[studentEmissionGroup, problemEmissionGroup];
        real slip_t = slip[studentEmissionGroup, problemEmissionGroup];
        int y = correctness[localStudentIdx, t];
        // P(correct=1|mastered) = 1-slip, P(correct=0|mastered) = slip
        real emit0 = y == 1 ? 1 - slip_t : slip_t; 
        // P(correct=1|not mastered) = guess, P(correct=0|not mastered) = 1-guess;
        real emit1 = y == 1 ? guess_t : 1 - guess_t;
        vector[2] filtered = to_vector([latent_mastery[1] * emit0, latent_mastery[2] * emit1]);
        real normalizer = sum(filtered);

        target_ += log(normalizer);
        filtered /= normalizer;

        if (t < L) {
          latent_mastery[1] = filtered[1] * (1 - learn_t) + filtered[2] * forget_t;
          latent_mastery[2] = filtered[1] * learn_t + filtered[2] * (1 - forget_t);
        }
      }

      localStudentIdx += 1;
    }

    return target_;
  }

  /**
  Helper function to apply priors for the parameters. If unif_prior is 0, applies the priors on the logit scale.
  Else stan will automatically apply uniform priors over the parameter space/support.
  */
    void handle_normal_priors_lp(int unif_prior,         // binary indicator for uniform prior
                                vector logit_param, // parameter on logit scale
                                vector prior_mu,  // prior means for normal distribution
                                vector prior_std // prior stds for normal distribution
                                ){
    if (unif_prior != 1) {
        // Apply normal priors on logit scale
        logit_param ~ normal(prior_mu, prior_std);
    }
    // else stan will automatically apply uniform over the parameter space/support.
  }
}


data {
    int<lower=1> nProblems;    // number of problems
    int<lower=1> nStudents;    // number of students

    // GROUPS
    // number of groups for students
    int<lower=1> nStudentGroupsInit;        // Initial state
    int<lower=1> nStudentGroupsTransition;  // Learn, forget
    int<lower=1> nStudentGroupsEmission;    // Guess, slip
    // number of groups for problems
    int<lower=1> nProblemGroupsTransition;  // Learn, forget
    int<lower=1> nProblemGroupsEmission;    // Guess, slip

    // group assignment for each student (1-based indexing)
    array[nStudents] int<lower=1, upper=nStudentGroupsInit> studentGroupsInit; 
    array[nStudents] int<lower=1, upper=nStudentGroupsTransition> studentGroupsTransition; 
    array[nStudents] int<lower=1, upper=nStudentGroupsEmission> studentGroupsEmission; 
    // group assignment for each problem (1-based indexing)
    array[nProblems] int<lower=1, upper=nProblemGroupsTransition> problemGroupsTransition; 
    array[nProblems] int<lower=1, upper=nProblemGroupsEmission> problemGroupsEmission; 

    // Ordered per-student interaction sequences.
    // The first interaction_lengths[student] entries are used; later entries are ignored.
    array[nStudents, nProblems] int<lower=-1, upper=1> correctness;
    array[nStudents, nProblems] int<lower=1, upper=nProblems> problem_sequence;
    array[nStudents] int<lower=1> interaction_lengths;
    
    // PRIORS
    // prior initial state
    row_vector[nStudentGroupsInit] prior_pi_know_mu;
    row_vector<lower=0>[nStudentGroupsInit] prior_pi_know_std;
    // prior transitions
    matrix[nStudentGroupsTransition, nProblemGroupsTransition] prior_learn_mu;
    matrix<lower=0>[nStudentGroupsTransition, nProblemGroupsTransition] prior_learn_std;
    matrix[nStudentGroupsTransition, nProblemGroupsTransition] prior_forget_mu;
    matrix<lower=0>[nStudentGroupsTransition, nProblemGroupsTransition] prior_forget_std;
    // prior emissions
    matrix[nStudentGroupsEmission, nProblemGroupsEmission] prior_guess_mu;
    matrix<lower=0>[nStudentGroupsEmission, nProblemGroupsEmission] prior_guess_std;
    matrix[nStudentGroupsEmission, nProblemGroupsEmission] prior_slip_mu;
    matrix<lower=0>[nStudentGroupsEmission, nProblemGroupsEmission] prior_slip_std;

    // UNIFORM PRIORS
    // binary indicator whether to use non-infomative uniform priors.
    int<lower=0, upper=1> unif_prior_pi_know;
    int<lower=0, upper=1> unif_prior_learn;
    int<lower=0, upper=1> unif_prior_forget;
    int<lower=0, upper=1> unif_prior_guess;
    int<lower=0, upper=1> unif_prior_slip;
    
}

parameters {
    // global initial logit for the know/masters state
    row_vector[nStudentGroupsInit] logit_pi_know_group;
    // group-level parameters (logit scale)
    matrix[nStudentGroupsTransition, nProblemGroupsTransition] logit_learn_group;
    matrix[nStudentGroupsTransition, nProblemGroupsTransition] logit_forget_group;
    matrix[nStudentGroupsEmission, nProblemGroupsEmission] logit_guess_group;
    matrix[nStudentGroupsEmission, nProblemGroupsEmission] logit_slip_group;
}

transformed parameters {    
    // group parameters in probability scale
    row_vector[nStudentGroupsInit] pi_know = inv_logit(logit_pi_know_group);
    // transitions
    matrix[nStudentGroupsTransition, nProblemGroupsTransition] learn = inv_logit(logit_learn_group);
    matrix[nStudentGroupsTransition, nProblemGroupsTransition] forget = inv_logit(logit_forget_group);
    // emissions (constrained to be <= 0.5)
    matrix[nStudentGroupsEmission, nProblemGroupsEmission] guess  = 0.5 * inv_logit(logit_guess_group);
    matrix[nStudentGroupsEmission, nProblemGroupsEmission] slip  = 0.5 * inv_logit(logit_slip_group);
}

model {
    // Bayesian Priors
    handle_normal_priors_lp(unif_prior_pi_know, to_vector(logit_pi_know_group), to_vector(prior_pi_know_mu), to_vector(prior_pi_know_std));
    handle_normal_priors_lp(unif_prior_learn, to_vector(logit_learn_group), to_vector(prior_learn_mu), to_vector(prior_learn_std));
    handle_normal_priors_lp(unif_prior_forget, to_vector(logit_forget_group), to_vector(prior_forget_mu), to_vector(prior_forget_std));
    handle_normal_priors_lp(unif_prior_guess, to_vector(logit_guess_group), to_vector(prior_guess_mu), to_vector(prior_guess_std));
    handle_normal_priors_lp(unif_prior_slip, to_vector(logit_slip_group), to_vector(prior_slip_mu), to_vector(prior_slip_std));

    // Parallelized likelihood computation
    int grainsize = 1; // use internal scheduler
    target += reduce_sum(partial_sum, correctness,
                        grainsize,
                        problem_sequence,
                        interaction_lengths,
                        learn, forget, guess, slip, to_vector(pi_know), // BKT params
                        studentGroupsInit,
                        studentGroupsTransition,
                        studentGroupsEmission,
                        problemGroupsTransition,
                        problemGroupsEmission
                        );

}



