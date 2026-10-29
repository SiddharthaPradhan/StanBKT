// NOTE: this file holds a single bare function definition, meant to be
// #include-d from inside another file's `functions { }` block (Stan does not
// allow more than one top-level `functions` block per compiled program).

/**
Computes P(know) at time T+1 for each student, i.e. after conditioning on all T
observed responses and applying one more learn/forget transition, via the
standard forward filter (same recursion as hidden_states.stan). Only the
terminal state is kept/returned. This is used for the post-test outcome regression.
*/
vector final_pknow_forward(array[] int lengths, // interaction_lengths, original student order
                            array[,] int correctness, // original student order (nStudents, nProblems)
                            vector learn_student, vector forget_student, // original student order
                            vector guess_student, vector slip_student,
                            vector pi_know_student) {
  int n = num_elements(lengths);
  vector[n] known = pi_know_student;
  for (student_idx in 1:n) {
      real k = known[student_idx];
      for (t in 1:lengths[student_idx]) {
          real num, denom;
          if (correctness[student_idx, t] == 1) {
              num = k * (1 - slip_student[student_idx]);
              denom = num + (1 - k) * guess_student[student_idx];
          } else {
              num = k * slip_student[student_idx];
              denom = num + (1 - k) * (1 - guess_student[student_idx]);
          }
          // guard against 0/0 during early sampling/optimization iterations, where
          // extreme unconstrained proposals can drive guess/slip to exactly 0 or 1
          real k_given_obs = num / fmax(denom, 1e-12);
          k = learn_student[student_idx] + k_given_obs * (1 - forget_student[student_idx] - learn_student[student_idx]);
      }
      known[student_idx] = k;
  }
  return known;
}
