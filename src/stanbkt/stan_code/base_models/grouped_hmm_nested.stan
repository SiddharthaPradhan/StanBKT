functions{

  /**
  Compute the partial sum for a slice of the log-omega matrixs.
  Slices are made along the outer array dimension (i.e. students).
  This version accounts for grouped HMMs with different A_matrices per group.
  */
  real partial_sum(array[] matrix logOmegaSlice,
                   int start, int end,
                   array[] matrix A_matrix_group,
                   vector pi, array[] int groups) {

    real target_ = 0.0;
    int localIdx = 1; // local index within the slice
    for(studentIdx in start:end){
        int studentGroupIdx = groups[studentIdx];
        target_ += hmm_marginal(logOmegaSlice[localIdx], A_matrix_group[studentGroupIdx], pi);
        localIdx += 1; // increment local index
    }
    return target_;
  }
}


data {
    int<lower=1> nProblems;    // number of problems
    int<lower=1> nStudents;    // number of students
    int<lower=1> nGroups;      // number of groups
    array[nStudents] int<lower=1, upper=nGroups> groups; // group assignment for each student
    array[nStudents, nProblems] int<lower=0, upper=1> correctness; // correctness matrix
    
}

parameters {
    // global initial logit for the know/masters state
    real logit_pi_know;

    // population-level parameters (on logit scale for better sampling)
    real logit_learn_population;
    real logit_forget_population;
    real logit_guess_population;  
    real logit_slip_population;
    
    // group-level parameters (logit scale)
    row_vector[nGroups] logit_learn_group_delta;
    row_vector[nGroups] logit_forget_group_delta;
    row_vector[nGroups] logit_guess_group_delta;
    row_vector[nGroups] logit_slip_group_delta;

}


transformed parameters {    
    real<lower=0, upper=1> pi_know_global = inv_logit(logit_pi_know);
  
    // Row-vector version of group delta + population parameters
    row_vector[nGroups] learn_vec = inv_logit(logit_learn_population + logit_learn_group_delta);
    row_vector[nGroups] forget_vec = inv_logit(logit_forget_population + logit_forget_group_delta);
    row_vector[nGroups] guess_vec = 0.5 * inv_logit(logit_guess_population + logit_guess_group_delta);
    row_vector[nGroups] slip_vec = 0.5 * inv_logit(logit_slip_population + logit_slip_group_delta);


}

model {
    // Bayesian Priors
    // Priors for population-level parameters
    logit_learn_population ~ normal(0, 5);
    logit_forget_population ~ normal(-2, 5);
    logit_guess_population ~ normal(-1, 5);   
    logit_slip_population ~ normal(-1, 5); 
    
    // Priors for group-level deviations (centered at 0 for shrinkage)
    logit_learn_group_delta ~ normal(0, 5);
    logit_forget_group_delta ~ normal(0, 5);
    logit_guess_group_delta ~ normal(0, 5);
    logit_slip_group_delta ~ normal(0, 5);
    
    // initial state distribution
    vector[2] pi = to_vector([1 - pi_know_global, pi_know_global]); 

    // group-level transition and emission variables
    // matrix form for HMM functions         
    array[nGroups] matrix[2, 2] A_matrix_group;
    array[nGroups] matrix[2, 2] B_matrix_group;

    for (group_idx in 1:nGroups) {
        // Convert to matrix form
        // Transition matrix A
        A_matrix_group[group_idx][1,] = [1 - learn_vec[group_idx], learn_vec[group_idx]];
        A_matrix_group[group_idx][2,] = [forget_vec[group_idx], 1 - forget_vec[group_idx]];
        // Emission matrix B
        B_matrix_group[group_idx][1,] = [1 - guess_vec[group_idx], guess_vec[group_idx]];
        B_matrix_group[group_idx][2,] = [slip_vec[group_idx], 1 - slip_vec[group_idx]];
    }

    array[nStudents] matrix[2, nProblems] log_omega;
    for(studentIdx in 1:nStudents){
        for(t in 1:nProblems) {
            for (state in 1:2) {
                // log P(correctness | hidden_state)
                int studentGroupIdx = groups[studentIdx];
                log_omega[studentIdx, state, t] = bernoulli_lpmf(correctness[studentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
            }
        }
    }
    
    // Likelihood
    // Original non-parallel code
    // for(studentIdx in 1:nStudents){
    //     int studentGroupIdx = groups[studentIdx];
    //     target += hmm_marginal(log_omega[studentIdx], A_matrix_group[studentGroupIdx], pi_population);
    // }
    // Parallelized likelihood computation
    int grainsize = 1; // default is 1, that uses an internal scheduler
    target += reduce_sum(partial_sum, log_omega,
                        grainsize,
                        A_matrix_group, pi, groups);
}

// generated quantities {
//     array[nStudents] matrix[2, nProblems] hidden_probs;
//     for (studentIdx in 1:nStudents) {
//         int studentGroupIdx = groups[studentIdx];
//         hidden_probs[studentIdx] = hmm_hidden_state_prob(log_omega[studentIdx],
//                             A_matrix_group[studentGroupIdx], pi_population);
//     }
// }

