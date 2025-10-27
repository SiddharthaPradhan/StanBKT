functions{
  /**
  Compute the partial sum for a slice of the log-omega matrixs.
  Slices are made along the outer array dimension (i.e. students).
  */  
real partial_sum(array[] matrix logOmegaSlice,
                   int start, int end,
                   matrix A_matrix,
                   vector pi) {

    real target_ = 0.0;
    int localIdx = 1;
    for(s in start:end){
        target_ += hmm_marginal(logOmegaSlice[localIdx], A_matrix, pi);
        localIdx += 1;
    }
    return target_;
  }
}

data {
    int<lower=1> nProblems;                      // number of problems
    int<lower=1> nStudents;                      // number of students
    int<lower=1> nSkills;                        // number of skills
    array[nStudents, nProblems] int<lower=0, upper=1> correctness; // correctness matrix
}

parameters {
    real pi_know_logit;
    real learn_logit;
    real forget_logit;
    real guess_logit;
    real slip_logit;    
}

transformed parameters {
    real<lower=0, upper=1> pi_know = inv_logit(pi_know_logit);
    real<lower=0, upper=1> learn = inv_logit(learn_logit);
    real<lower=0, upper=1> forget = inv_logit(forget_logit);
    real<lower=0, upper=0.5> guess = 0.5 * inv_logit(guess_logit);
    real<lower=0, upper=0.5> slip = 0.5 * inv_logit(slip_logit);
 
}

model {
    // Default Priors
    learn_logit ~ normal(0, 5);
    forget_logit ~ normal(-2, 5);
    guess_logit ~ normal(-1, 5);
    slip_logit ~ normal(-1, 5);
    // pi_know_logit ~ normal(-2, 5);


    vector[2] pi = to_vector([1 - pi_know, pi_know]); // initial state distribution

    matrix[2, 2] A_matrix;
    matrix[2, 2] B_matrix;
    A_matrix[1,] = [1 - learn, learn];
    A_matrix[2,] = [forget, 1 - forget];
    B_matrix[1,] = [1 - guess, guess];
    B_matrix[2,] = [slip, 1 - slip];

    array[nStudents] matrix[2, nProblems] log_omega;
    // TODO try to parallelize this using map_rect
    for(s in 1:nStudents){
        for(t in 1:nProblems) {
            for (state in 1:2) {
                log_omega[s, state, t] = bernoulli_lpmf(correctness[s, t] | B_matrix[state, 2]);
            }
        }
    }
    // Original non-parallel code
    // for(s in 1:nStudents){
        //     target += hmm_marginal(log_omega[s], A_matrix, pi);
        // }
        
        // Parallelized code
    int grainsize = 1; // default is 1, that uses an internal scheduler
    target += reduce_sum(partial_sum, log_omega,
                        grainsize,
                        A_matrix, pi);

}

// generated quantities {
//     array[nStudents] matrix[2, nProblems] hidden_probs;
//     for (s in 1:nStudents) {
//         hidden_probs[s] = hmm_hidden_state_prob(log_omega[s], A_matrix, pi);
//     }
// }

