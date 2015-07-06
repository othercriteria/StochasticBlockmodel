#define MAIN  // (this suppresses the definitions of main in count.c and sample.c)

#include "count.c"
#include "sample.c"



int main(int argc, char** argv) {
    int m = N_ROWS;
    int n = N_COLS;
    
    // Convert observed_matrix to int**
    int** A; allocate_matrix_copy(int,A,observed_matrix,m,n);
    
    // compute row and column sums
    int* p; allocate_vector(int,p,m);
    int* q; allocate_vector(int,q,n);
    for (int i = 0; i<m; i++) {
        p[i] = 0;
        for (int j = 0; j<n; j++) p[i] += A[i][j];
    }
    for (int j = 0; j<n; j++) {
        q[j] = 0;
        for (int i = 0; i<m; i++) q[j] += A[i][j];
    }

    // Initialize counting variables
    data_t* v = initialize_variables(p,q,m,n,"");
        
        
    // COUNTING =====================================
    
    // Count the number of matrices
    print("Counting...\n");
    bigint* result = count(v,matrix_type);
    gmp_printf("Number of matrices = %Zd\n",*result); 
    
    
    // SAMPLING =====================================
    
    // Allocate memory for the matrix and the statistics histogram
    int** matrix; allocate_matrix(int,matrix,m,n);
    int max_S = max_statistic(m,n);
    int* histogram; allocate_vector(int,histogram,max_S+1);
    
    // Start the clock (to time how long it takes to run)
    clock_t start_time = clock();
    
    // Sample
    int minimum = max_S;
    int maximum = 0;
    print("Sampling...\n");
    for (int i = 0; i<n_samples; i++) {
        // draw a sample matrix
        sample(matrix,v,matrix_type);
        
        // compute test statistic
        int S_i = compute_statistic(matrix,p,q,m,n);
        
        // update the histogram
        histogram[S_i] += 1;
        
        // Update minimum and maximum
        minimum = min(minimum,S_i);
        maximum = max(maximum,S_i);
    }
    // Stop the clock
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time))/CLOCKS_PER_SEC;
    
    // Compute the test statistic on the observed matrix
    int S = compute_statistic(A,p,q,m,n);
    
    // Estimate p-value
    int count = 0;
    for (int i = S; i<=max_S; i++) count += histogram[i];
    double p_value = ((double)count)/n_samples;
    
    // Display results
    printf("\n============== Histogram ==============\n");
    for (int i = minimum; i<=maximum; i++) printf("%d: %d\n",i,histogram[i]);
    printf("\n============== Estimated p-value ==============\n");
    printf("p-value = %f\n",p_value);
    printf("\n============== Timing ==============\n");
    print("Elapsed CPU time = %f seconds\n",elapsed_time);
    print("CPU time per sample = %f seconds\n",elapsed_time/n_samples);
    
    return 0;
}












