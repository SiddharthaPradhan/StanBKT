data {
    int<lower=1> nProblems;    // number of problems
    int<lower=1> nStudents;    // number of students
    int<lower=1> nGroups;      // number of groups
    array[nStudents] int<lower=1, upper=nGroups> groups; // group assignment for each student (1-based indexing)
    array[nStudents, nProblems] int<lower=0, upper=1> correctness; // correctness matrix
    // parameters from the fitted model
    row_vector[nGroups] pi_know;
    row_vector[nGroups] learn;
    row_vector[nGroups] forget;
    row_vector[nGroups] guess;
    row_vector[nGroups] slip;
}