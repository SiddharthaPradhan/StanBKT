functions{

  /**
  Compute the log density for a slice of students.
  Slices are made along the outer array dimension (i.e. students).
  This version accounts for grouped HMMs with different transitions (A_matrix_group), 
  emmisions (B_matrix_group) and priors (pi) per group.
  */
  real partial_sum(array[,] int correctness, // sliced correctness matrix (nStudentsInSlice, nProblems) 
                   int start, int end, // slice indexes
                   int nProblems, 
                   array[] matrix A_matrix_group, // Transition matrices (nGroups, 2, 2)
                   array[] matrix B_matrix_group, // Emission matrices (nGroups, 2, 2)
                   array[]vector pi, // Initial state distributions (nGroups, 2)
                   array[] int groups // group assignment for each student (nStudents)
                   ) {

    real target_ = 0.0; // accumulator for the log density
    // Compute log emission probabilities log P(correctness | hidden_state)
    int localStudentIdx = 1; // local index within the slice; studentIdx refers to global index (non-sliced)
    for(studentIdx in start:end){
        matrix[2, nProblems] logOmegaStudent;
        int studentGroupIdx = groups[studentIdx];
        for(t in 1:nProblems) {
            for (state in 1:2) {
                // log P(correctness | hidden_state)
                // TODO left off here, test if lupmf is correct here. Previously was lpmf
                // access correctness matrix for student studentIdx using the localStudentIdx
                logOmegaStudent[state, t] = bernoulli_lpmf(correctness[localStudentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
            }
        }
        // marginalize out the hidden states for this student and add to target
        target_ += hmm_marginal(logOmegaStudent, A_matrix_group[studentGroupIdx], pi[studentGroupIdx]);
        localStudentIdx += 1; // increment local index
    }
    return target_;
  }
}


data {
    int<lower=1> nProblems;    // number of problems
    int<lower=1> nStudents;    // number of students
    int<lower=1> nGroups;      // number of groups
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
    row_vector[nGroups] guess  = 0.5 * inv_logit(logit_guess_group);
    row_vector[nGroups] slip  = 0.5 * inv_logit(logit_slip_group);
}

model {
    // Bayesian Priors
    // TODO implement user input for the priors. This will add 10 parameters (mean and sd for each of the 5 parameter types).
    logit_pi_know_group ~ normal(-2, 5);
    logit_learn_group ~ normal(0, 5);
    logit_forget_group ~ normal(-2, 5);
    logit_guess_group ~ normal(-1, 5);
    logit_slip_group ~ normal(-1, 5);

    // The following variables are based on the parameters and converted into suitable vector or matrix form.
    // While this could have been technically been done in the transformed parameters block, 
    // it causes severe memory overhead, both while sampling and saving the fitted model. 
    // Declaring the parameters here throws away these intermediate values after using them for estimation.

    array[nGroups] vector[2] pi;

    // group-level transition and emission variables
    // matrix form for HMM functions         
    array[nGroups] matrix[2, 2] A_matrix_group;
    array[nGroups] matrix[2, 2] B_matrix_group;

    for (group_idx in 1:nGroups) {
        pi[group_idx] = to_vector([1 - pi_know[group_idx], pi_know[group_idx]]);
        // Convert to matrix form
        // Transition matrix A
        A_matrix_group[group_idx][1,] = [1 - learn[group_idx], learn[group_idx]];
        A_matrix_group[group_idx][2,] = [forget[group_idx], 1 - forget[group_idx]];
        // Emission matrix B
        B_matrix_group[group_idx][1,] = [1 - guess[group_idx], guess[group_idx]];
        B_matrix_group[group_idx][2,] = [slip[group_idx], 1 - slip[group_idx]];
    }

    // Parallelized likelihood computation
    int grainsize = 1; // default is 1, that uses an internal scheduler
    target += reduce_sum(partial_sum, correctness,
                        grainsize,
                        nProblems,
                        A_matrix_group, B_matrix_group, 
                        pi, groups);

}



// We need to recompute log_omega for here as we use reduce_sum in the model which does not retain it.
// Note on computation: This is perfectly fine as the GQ block runs once per iteration, while the model block runs several times per iteration.
//                      The added overhead is neglible in comparison to not using reduce_sum and parallelizing the model block.


// TODO: compute LOO for model comparison.
generated quantities {    
    // Smoothed marginal probability of the "know" state at each time step
    array[nStudents] row_vector[nProblems] know_probs; 
    // Most likely hidden state at each time step (1 = not know, 2 = know)
    array[nStudents, nProblems] int most_likely_state;
    // Probability of the most likely hidden state at each time step
    array[nStudents, nProblems] real most_likely_state_prob;
    // Log posterior probability of the most likely state sequence p(x | y, theta)
    array[nStudents] real most_likely_sequence_log_p;

    // create local scope to avoid saving intermediate variables
    {
        array[nGroups] vector[2] pi;
        // group-level transition and emission variables
        // matrix form for HMM functions         
        array[nGroups] matrix[2, 2] A_matrix_group;
        array[nGroups] matrix[2, 2] B_matrix_group;

        for (group_idx in 1:nGroups) {
            pi[group_idx] = to_vector([1 - pi_know[group_idx], pi_know[group_idx]]);
            // Transition matrix A
            A_matrix_group[group_idx][1,] = [1 - learn[group_idx], learn[group_idx]];
            A_matrix_group[group_idx][2,] = [forget[group_idx], 1 - forget[group_idx]];
            // Emission matrix B
            B_matrix_group[group_idx][1,] = [1 - guess[group_idx], guess[group_idx]];
            B_matrix_group[group_idx][2,] = [slip[group_idx], 1 - slip[group_idx]];
        }

        for (studentIdx in 1:nStudents) {
            matrix[2, nProblems] logOmegaStudent;
            matrix[2, nProblems] state_probs; // smoothed P(z_t | y_1:T)
            // Viterbi dynamic programming tables
            matrix[2, nProblems] delta;       // log p(y_{1:t}, x_t = i | theta) along best path
            array[2, nProblems] int backpointer; // backpointers for Viterbi
            array[nProblems] int viterbi_path;   // MAP state sequence x*
            int studentGroupIdx = groups[studentIdx];

            // Compute log emission probabilities log P(correctness | hidden_state)
            for (t in 1:nProblems) {
                for (state in 1:2) {
                    logOmegaStudent[state, t] = bernoulli_lpmf(correctness[studentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
                }
            }

            // Forward-backward (smoothed) state probabilities
            state_probs = hmm_hidden_state_prob(logOmegaStudent,
                                                A_matrix_group[studentGroupIdx],
                                                pi[studentGroupIdx]);

            // Store smoothed P(know) marginals
            know_probs[studentIdx] = state_probs[2];

            // Viterbi algorithm to find MAP state sequence x*
            // Initialization
            for (state in 1:2) {
                delta[state, 1] = log(pi[studentGroupIdx][state]) +
                                  logOmegaStudent[state, 1];
                backpointer[state, 1] = 0;
            }

            // Recursion
            for (t in 2:nProblems) {
                for (state_to in 1:2) {
                    real best_val = negative_infinity();
                    int arg_max = 1;
                    for (state_from in 1:2) {
                        real cand = delta[state_from, t - 1] +
                                    log(A_matrix_group[studentGroupIdx][state_from, state_to]);
                        if (cand > best_val) {
                            best_val = cand;
                            arg_max = state_from;
                        }
                    }
                    delta[state_to, t] = best_val + logOmegaStudent[state_to, t];
                    backpointer[state_to, t] = arg_max;
                }
            }

            // Termination: best final state
            int last_state = 1;
            real best_last = delta[1, nProblems];
            if (delta[2, nProblems] > best_last) {
                last_state = 2;
                best_last = delta[2, nProblems];
            }

            // Backtrack to recover Viterbi path x*
            viterbi_path[nProblems] = last_state;
            for (t in 1:(nProblems - 1)) {
                int tt = nProblems - t;
                viterbi_path[tt] = backpointer[viterbi_path[tt + 1], tt + 1];
            }

            // Store path and its per-time marginal probabilities
            for (t in 1:nProblems) {
                most_likely_state[studentIdx, t] = viterbi_path[t];
                most_likely_state_prob[studentIdx, t] =
                    state_probs[most_likely_state[studentIdx, t], t];
            }

            // Compute log posterior probability of the MAP sequence:
            // log p(x* | y, theta) = log p(x*, y | theta) - log p(y | theta)
            real log_evidence = hmm_marginal(logOmegaStudent,
                                             A_matrix_group[studentGroupIdx],
                                             pi[studentGroupIdx]);
            most_likely_sequence_log_p[studentIdx] = best_last - log_evidence;
        }
    }
}

// TODO: Check if we can use reduce-sum in gernated quantities to compute the log_omega
    // // Compute log emission probabilities log P(correctness | hidden_state)
    // array[nStudents] matrix[2, nProblems] log_omega;
    // for(studentIdx in 1:nStudents){
    //     for(t in 1:nProblems) {
    //         for (state in 1:2) {
    //             // log P(correctness | hidden_state)
    //             int studentGroupIdx = groups[studentIdx];
    //             // TODO: check if lupmf is correct here. Previously was lpmf
    //             log_omega[studentIdx, state, t] = bernoulli_lupmf(correctness[studentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
    //         }
    //     }
    // } 
