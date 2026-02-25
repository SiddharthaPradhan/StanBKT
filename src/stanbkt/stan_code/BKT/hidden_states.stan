// include the base BKT model
#include "BKT_model.stan"


// Run forward to compute the probability of knowing/mastery and correctness.
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
