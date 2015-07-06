// Patterson.c
// Estimate p-value for Patterson and Atmar's montane mammals dataset,
// using their nested subsets statistic.

#include "utilities.h"

// parameters
int n_samples = 100000;
int matrix_type = 0;

// observed matrix
#define N_ROWS 26
#define N_COLS 28
int observed_matrix[N_ROWS][N_COLS] = {
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,0,0},
    {1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,0,0,0},
    {1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,0,1,1,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0},
    {1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,0,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,0,1,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}};

// Return an upper bound for the test statistic on a m-by-n matrix.
// The minimum possible value is assumed to be 0.
int max_statistic(int m, int n) {
    return m*n;
}

// Compute the test statistic for a m-by-n matrix.
// (Must return an integer between 0 and max_statistic(m,n).)
int compute_statistic(int** matrix, int* p, int* q, int m, int n) {
    // This function computes Patterson and Atmar's nested subset statistic.
    
    // Find the smallest habitat for each species
    int* smallest; allocate_vector(int,smallest,m);
    for (int i = 0; i<m; i++) {
        smallest[i] = m;
        for (int j = 0; j<n; j++) {
            if (matrix[i][j]==1) smallest[i] = min(smallest[i],q[j]);
        }
    }
    
    // Compute statistic
    int s = 0;
    for (int i = 0; i<m; i++) {
        for (int j = 0; j<n; j++) {
            if (q[j]>smallest[i]) s += 1-matrix[i][j];
        }
    }
    return s;
}


// Import main() for computing p-values
#include "p_value.c"







