// Darwin.c
// Estimate p-value for Darwin's finches dataset, using Roberts & Stone (1990)'s $\bar S^2$ statistic.

// Got p-value = 0.000463 for n_samples = 10000000

// parameters
int n_samples = 100000;
int matrix_type = 0;

// observed matrix
#define N_ROWS 13
#define N_COLS 17
int observed_matrix[N_ROWS][N_COLS] = {
    {0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1},
    {1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,0,0},
    {1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0},
    {0,0,1,1,1,0,0,1,0,1,0,1,1,0,1,1,1},
    {1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0},
    {0,0,1,1,1,1,1,1,1,0,0,1,0,1,1,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0},
    {0,0,1,1,1,1,1,1,1,1,0,1,0,0,1,0,0},
    {0,0,1,1,1,1,1,1,1,1,0,1,0,1,1,0,0},
    {0,0,1,1,1,0,1,1,0,1,0,0,0,0,0,0,0},
    {0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1}};

// Return an upper bound for the test statistic on a m-by-n matrix.
// The minimum possible value is assumed to be 0.
int max_statistic(int m, int n) {
    return n*n*(m*(m-1))/2;
}

// Compute the test statistic for a m-by-n matrix.
// (Must return an integer between 0 and max_statistic(m,n).)
int compute_statistic(int** matrix, int* p, int* q, int m, int n) {
    // This function computes nchoosek(m,2) times $\bar S^2$.
    int s = 0;
    for (int i = 0; i<m-1; i++) {
        for (int j = i+1; j<m; j++) {
            int C_ij = 0;
            for (int k = 0; k<n; k++) {
                C_ij += matrix[i][k]*matrix[j][k];
            }
            s += C_ij*C_ij;
        }
    }
    return s;
}


// Import main() for computing p-values
#include "p_value.c"







