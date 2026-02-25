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
    array[nStudents] row_vector[nProblems] pKnow; // P(know/mastery | all previous responses) for each student and problem
    array[nStudents] row_vector[nProblems] pCorrectness; // P(correctness | all previous responses) for each student and problem
    // create local scope to avoid saving intermediate variables
    {
        for (studentIdx in 1:nStudents) {
            int studentGroupIdx = groups[studentIdx];
            real studentInit = pi_know[studentGroupIdx];  // initial probability of knowing
            real studentLearn = learn[studentGroupIdx];   // transition probability of learning
            real studentGuess = guess[studentGroupIdx];   // emission probability of guessing
            real studentSlip = slip[studentGroupIdx];     // emission probability of slipping
            // Precompute 1 - parameter values to save computation in the loop
            real oneMinusStudentSlip = 1.0 - studentSlip;
            real oneMinusStudentGuess = 1.0 - studentGuess;
            real oneMinusStudentForget = 1.0 - forget[studentGroupIdx];
            pKnow[studentIdx, 1] = studentInit; // P(know/mastery) for the first problem is just the prior
            pCorrectness[studentIdx, 1] = studentInit * oneMinusStudentSlip + (1 - studentInit) * studentGuess; 
            for (t in 1:(nProblems - 1)) {    
                real currentProbKnow = pKnow[studentIdx, t];
                // update the probability of knowing/mastering based on the current response 
                // P(know/mastery | response) = P(response | know/mastery) * P(know/mastery) / P(response)
                real numerator, denominator;
                if (correctness[studentIdx, t]) {
                    numerator = currentProbKnow * oneMinusStudentSlip;
                    denominator = numerator + (1.0 - currentProbKnow) * studentGuess;
                } else {
                    numerator = currentProbKnow * studentSlip;
                    denominator = numerator + (1.0 - currentProbKnow) * oneMinusStudentGuess;
                }
                real pKnowGivenObs = numerator / denominator;
                // update the probability for knowing/mastering
                pKnow[studentIdx, t + 1] = pKnowGivenObs * oneMinusStudentForget + (1 - pKnowGivenObs) * studentLearn;
                pCorrectness[studentIdx, t + 1] = pKnow[studentIdx, t + 1] * oneMinusStudentSlip + 
                                                  (1 - pKnow[studentIdx, t + 1]) * studentGuess;


            }
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
