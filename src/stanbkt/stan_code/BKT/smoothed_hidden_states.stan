// include the base BKT model
#include "BKT_model.stan"

// Run forward-backward to compute the smoothed probability of knowing/mastery for each student and problem.
generated quantities {    
    array[nStudents] row_vector[nProblems] pKnow; // P(know/mastery | all previous responses) for each student and problem
    // create local scope to avoid saving intermediate variables
    {
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

        for (studentIdx in 1:nStudents) {
            // TOOD: the nProblems will vary by student, we need to pass in a lens array for nProblems
            // Python will keep track of the id to problem_id mapping
            matrix[2, interaction_lengths[studentIdx]] logOmegaStudent;
            int studentGroupIdx = groups[studentIdx];
            for(t in 1:interaction_lengths[studentIdx]) {
                for (state in 1:2) {
                    // log P(correctness | hidden_state)
                    logOmegaStudent[state, t] = bernoulli_lpmf(correctness[studentIdx, t] | B_matrix_group[studentGroupIdx, state, 2]);
                }
            }
            int paddingNALength = nProblems - interaction_lengths[studentIdx];
            pKnow[studentIdx, 1:interaction_lengths[studentIdx]] = hmm_hidden_state_prob(logOmegaStudent, A_matrix_group[studentGroupIdx], pi[studentGroupIdx])[2];
            pKnow[studentIdx, interaction_lengths[studentIdx]+1:nProblems] = rep_row_vector(-1.0, paddingNALength);
        }
    }
}
