// include the base BKT model
#include "BKT_model.stan"

// We need to recompute log_omega for here as we use reduce_sum in the model which does not retain it.
// Note on computation: This is perfectly fine as the GQ block runs once per iteration, while the model block runs several times per iteration.
//                      The added overhead is neglible in comparison to not using reduce_sum and parallelizing the model block.


// Run forward-backward to compute the smoothed probability of knowing/mastery for each student and problem.
generated quantities {    
    array[nStudents] row_vector[nProblems] pKnow;        // P(know/mastery | all observations) for each student and problem
    array[nStudents] row_vector[nProblems] pCorrectness; // P(correct at t | all observations) = f(pKnow[t+1])
    // create local scope to avoid saving intermediate variables
    {
        
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
        array[individual_pi_know == 1 ? nStudents : nGroups] vector[2] pi;
        if (individual_pi_know == 1) {
            for (student_idx in 1:nStudents) {
                pi[student_idx] = to_vector([1 - pi_know[student_idx], pi_know[student_idx]]);
            }
        } else {
            for (group_idx in 1:nGroups) {
                pi[group_idx] = to_vector([1 - pi_know[group_idx], pi_know[group_idx]]);
            }
        }
        
        for (studentIdx in 1:nStudents) {
            matrix[2, interaction_lengths[studentIdx]] logOmegaStudent;
            int studentGroupIdx = groups[studentIdx];
            for(t in 1:interaction_lengths[studentIdx]) {
                for (state in 1:2) {
                    // log P(correctness | hidden_state)
                    logOmegaStudent[state, t] = bernoulli_lpmf(correctness[studentIdx, t] |  B_matrix_group[studentGroupIdx, state, 2]);
                }
            }
            int L = interaction_lengths[studentIdx];
            int paddingNALength = nProblems - L;
            pKnow[studentIdx, 1:L] = hmm_hidden_state_prob(logOmegaStudent, 
                                            A_matrix_group[studentGroupIdx], 
                                            pi[individual_pi_know == 1 ? studentIdx : studentGroupIdx])[2];
            pKnow[studentIdx, L+1:nProblems] = rep_row_vector(-1.0, paddingNALength);

            // pCorrectness[t] = pKnow[t] * (1 - slip) + (1 - pKnow[t]) * guess
            real oneMinusStudentSlip = 1.0 - slip[studentGroupIdx];
            real studentGuess = guess[studentGroupIdx];
            pCorrectness[studentIdx, 1:L] = pKnow[studentIdx, 1:L] * oneMinusStudentSlip
                                                + (1 - pKnow[studentIdx, 1:L]) * studentGuess;
            pCorrectness[studentIdx, L+1:nProblems] = rep_row_vector(-1.0, paddingNALength);
        }
    }
}
