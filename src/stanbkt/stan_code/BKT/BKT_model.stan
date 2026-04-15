functions{
  /**
  Compute the log density for a slice of students.
  Slices are made along the outer array dimension (i.e. students).
  This version accounts for grouped HMMs with different transitions (A_matrix_group), 
  emmisions (B_matrix_group) and priors (pi) per group.
  */
  real partial_sum(array[,] int correctness, // sliced correctness matrix (nStudentsInSlice, nProblems) 
                   int start, int end, // slice indexes
                   array[] int interaction_lengths, // lengths of each student's interaction sequence (nStudents)
                   array[] matrix A_matrix_group, // Transition matrices (nGroups, 2, 2)
                   array[] matrix B_matrix_group, // Emission matrices (nGroups, 2, 2)
                   array[]vector pi, // Initial state distributions (nGroups, 2)
                   array[] int groups, // group assignment for each student (nStudents)
                   int individual_pi_know // binary indicator for individualized pi_know (scalar) 
                   ) {

    real target_ = 0.0; // accumulator for the log density
    // Compute log emission probabilities log P(correctness | hidden_state)
    int localStudentIdx = 1; // local index within the slice; studentIdx refers to global index (non-sliced)
    for(studentIdx in start:end){
        matrix[2, interaction_lengths[studentIdx]] logOmegaStudent;
        int studentGroupIdx = groups[studentIdx];
        for(t in 1:interaction_lengths[studentIdx]) {
            for (state in 1:2) {
                // log P(correctness | hidden_state)
                // access correctness matrix for student studentIdx using the localStudentIdx
                logOmegaStudent[state, t] = bernoulli_lpmf(correctness[localStudentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
            }
        }
        // marginalize out the hidden states for this student and add to target
        target_ += hmm_marginal(logOmegaStudent, A_matrix_group[studentGroupIdx], pi[individual_pi_know == 1 ? studentIdx : studentGroupIdx]);
        localStudentIdx += 1; // increment local index
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
}

parameters {
    // global initial logit for the know/masters state
    row_vector[individual_pi_know == 1 ? nStudents : nGroups] logit_pi_know_group;
    // group-level parameters (logit scale)
    row_vector[nGroups] logit_learn_group;
    row_vector[nGroups] logit_forget_group;
    row_vector[nGroups] logit_guess_group;
    row_vector[nGroups] logit_slip_group;
}

transformed parameters {    
    // group parameters in probability scale
    row_vector[individual_pi_know == 1 ? nStudents : nGroups] pi_know = inv_logit(logit_pi_know_group);
    row_vector[nGroups] learn = inv_logit(logit_learn_group);
    row_vector[nGroups] forget = inv_logit(logit_forget_group);
    // Constrain guess and slip to be <= 0.5
    row_vector[nGroups] guess  = 0.5 * inv_logit(logit_guess_group);
    row_vector[nGroups] slip  = 0.5 * inv_logit(logit_slip_group);
}

model {
    // Bayesian Priors
    handle_normal_priors_lp(unif_prior_pi_know, logit_pi_know_group, prior_pi_know_mu, prior_pi_know_std);
    handle_normal_priors_lp(unif_prior_learn, logit_learn_group, prior_learn_mu, prior_learn_std);
    handle_normal_priors_lp(unif_prior_forget, logit_forget_group, prior_forget_mu, prior_forget_std);
    handle_normal_priors_lp(unif_prior_guess, logit_guess_group, prior_guess_mu, prior_guess_std);
    handle_normal_priors_lp(unif_prior_slip, logit_slip_group, prior_slip_mu, prior_slip_std);


    // The following variables are based on the parameters and converted into suitable vector or matrix form.
    // While this could have been technically been done in the transformed parameters block, 
    // it causes severe memory overhead, both while sampling and saving the fitted model. 
    // Declaring the parameters here throws away these intermediate values after using them for estimation.

    array[individual_pi_know == 1 ? nStudents : nGroups] vector[2] pi;

    // group-level transition and emission variables
    // matrix form for HMM functions         
    array[nGroups] matrix[2, 2] A_matrix_group;
    array[nGroups] matrix[2, 2] B_matrix_group;

    for (group_idx in 1:nGroups) {
        // Convert to matrix form
        // Transition matrix A
        A_matrix_group[group_idx][1,] = [1 - learn[group_idx], learn[group_idx]];
        A_matrix_group[group_idx][2,] = [forget[group_idx], 1 - forget[group_idx]];
        // Emission matrix B
        B_matrix_group[group_idx][1,] = [1 - guess[group_idx], guess[group_idx]];
        B_matrix_group[group_idx][2,] = [slip[group_idx], 1 - slip[group_idx]];
    }

    // initial state distribution pi
    if (individual_pi_know == 1) {
        for (student_idx in 1:nStudents) {
            pi[student_idx] = to_vector([1 - pi_know[student_idx], pi_know[student_idx]]);
        }
    } else {
        for (group_idx in 1:nGroups) {
            pi[group_idx] = to_vector([1 - pi_know[group_idx], pi_know[group_idx]]);
        }
    }

    // Parallelized likelihood computation
    int grainsize = 1; // use internal scheduler
    target += reduce_sum(partial_sum, correctness,
                        grainsize,
                        interaction_lengths,
                        A_matrix_group, B_matrix_group, 
                        pi, groups, individual_pi_know);

}



