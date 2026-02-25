functions{

  /**
  Compute the partial sum for a slice of the log-omega matrixs.
  Slices are made along the outer array dimension (i.e. students).
  This version accounts for grouped HMMs with different A_matrices per group.
  */
  real partial_sum(array[] int student_indexes_slice, // student indexes for the slice
                   int start, int end, // slice indexes
                   int nProblems, 
                   array[,] int correctness, // correctness matrix (nStudents, nProblems) 
                   // Each KC and group has its own transition and emission matrices.
                   array[] matrix A_matrix_group, // Transition matrices (nGroups, 2, 2)
                   array[] matrix B_matrix_group, // Emission matrices (nGroups, 2, 2)
                   array[]vector pi, // Initial state distributions (nGroups, 2)
                   array[] int groups, // group assignment for each student (nStudents)
                   array[] int kcSequence // KC assignment for each problem (nProblems)
                   ) {

    real target_ = 0.0;
    // Compute log emission probabilities log P(correctness | hidden_state)
    int localIdx = 1; // local index within the slice; studentIdx refers to global index (non-sliced)
    // array size: number of students in the slice, indexed by localIdx
    for(studentIdx in start:end){
        matrix[2, nProblems] logOmegaSlice;
        for(t in 1:nProblems) {
            for (state in 1:2) {
                // log P(correctness | hidden_state)
                int studentGroupIdx = groups[studentIdx];
                // TODO left off here, test if lupmf is correct here. Previously was lpmf
                int problemKcIdx = kcSequence[t];
                // AND select B_matrix by problemKcIdx
                logOmegaSlice[state, t] = bernoulli_lpmf(correctness[studentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
            }
        }
        int studentGroupIdx = groups[studentIdx];
        // marginalize out the hidden states for this student and add to target
        target_ += hmm_marginal(logOmegaSlice, A_matrix_group[studentGroupIdx], pi);
        localIdx += 1; // increment local index
    }
    return target_;
  }
}


data {
    int<lower=1> nProblems;    // number of problems
    int<lower=1> nStudents;    // number of students
    int<lower=1> nGroups;      // number of groups
    int<lower=1> nKcs;         // number of knowledge components
    array[nProblems] int<lower=1, upper=nKcs> kcSequence; // KC assignment for each problem (1-based indexing)
    array[nStudents] int<lower=1, upper=nGroups> groups; // group assignment for each student (1-based indexing)
    array[nStudents, nProblems] int<lower=0, upper=1> correctness; // correctness matrix
}

parameters {
    // global initial logit for the know/masters state
    row_vector[nGroups] logit_pi_know_group;
    // group-level parameters (logit scale)
    row_vector[nGroups] logit_learn_group;
    row_vector[nGroups] logit_forget_group;
    row_vector[nGroups] logit_guess_group;
    row_vector[nGroups] logit_slip_group;
}


transformed parameters {    
    // group parameters in probability scale
    row_vector[nGroups] pi_know = inv_logit(logit_pi_know_group);
    row_vector[nGroups] learn = inv_logit(logit_learn_group);
    row_vector[nGroups] forget = inv_logit(logit_forget_group);
    // Constrain guess and slip to be <= 0.5
    row_vector[nGroups] guess  = 0.5 * inv_logit(logit_guess_group[kc]);
    row_vector[nGroups] slip  = 0.5 * inv_logit(logit_slip_group[kc]);
}

model {
    // Bayesian Priors
    // TODO implement user input for the priors. This will add 10 parameters (mean and sd for each of the 5 parameter types).
    for (kc in 1:nKcs) {
        logit_pi_know_group[kc] ~ normal(-2, 5);
        logit_learn_group[kc] ~ normal(0, 5);
        logit_forget_group[kc] ~ normal(-2, 5);
        logit_guess_group[kc] ~ normal(-1, 5);
        logit_slip_group[kc] ~ normal(-1, 5);
    }
        
    // initial state distribution
    array[nGroups] vector[2] pi;


    // group-level transition and emission variables
    // matrix form for HMM functions         
    array[nGroups] matrix[2, 2] A_matrix_group;
    array[nGroups] matrix[2, 2] B_matrix_group;

    for (group_idx in 1:nGroups) {
        pi[group_idx] = to_vector([1 - pi_know[kc, group_idx], pi_know[kc, group_idx]]);
        // Convert to matrix form
        // Transition matrix A
        A_matrix_group[kc, group_idx][1,] = [1 - learn[kc, group_idx], learn[kc, group_idx]];
        A_matrix_group[kc, group_idx][2,] = [forget[kc, group_idx], 1 - forget[kc, group_idx]];
        // Emission matrix B
        B_matrix_group[kc, group_idx][1,] = [1 - guess[kc, group_idx], guess[kc, group_idx]];
        B_matrix_group[kc, group_idx][2,] = [slip[kc, group_idx], 1 - slip[kc, group_idx]];
    }
    
    // Compute log emission probabilities log P(correctness | hidden_state)
    array[nStudents] matrix[2, nProblems] log_omega;
    for(studentIdx in 1:nStudents){
        for(t in 1:nProblems) {
            for (state in 1:2) {
                // log P(correctness | hidden_state)
                int studentGroupIdx = groups[studentIdx];
                int kc = kcSequence[t];
                // TODO: check if lupmf is correct here. Previously was lpmf
                log_omega[studentIdx, state, t] = bernoulli_lupmf(correctness[studentIdx, t] | B_matrix_group[kc, studentGroupIdx, state, 2]);
            }
        }
    }  

   
    // Parallelized likelihood computation
    int grainsize = 1; // default is 1, that uses an internal scheduler
    target += reduce_sum(partial_sum, log_omega,
                        grainsize,
                        A_matrix_group, pi, groups);
}

// We need to recompute log_omega for here as we use reduce_sum in the model which does not retain it.
// Note on computation: This is perfectly fine as the GQ block runs once per iteration, while the model block runs several times per iteration.
//                      The added overhead is neglible in comparison to not using reduce_sum and parallelizing the model block.

// TODO: compute LOO for model comparison.
// generated quantities {
//     array[nStudents] matrix[2, nProblems] hidden_probs;
//     for (studentIdx in 1:nStudents) {
//         int studentGroupIdx = groups[studentIdx];
//         hidden_probs[studentIdx] = hmm_hidden_state_prob(log_omega[studentIdx],
//                             A_matrix_group[studentGroupIdx], pi_population);
//     }
// }

