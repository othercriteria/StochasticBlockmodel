#include "common.h"
#include "count_recursion.c"


void validate_margins(int* p, int* q, int m, int n) {
    int p_sum = 0;
    for (int i = 0; i<m; i++) {
        assert(p[i]>=0, "All row and column sums must be nonnegative.");
        p_sum += p[i];
    }
    int q_sum = 0;
    for (int i = 0; i<n; i++) {
        assert(q[i]>=0, "All row and column sums must be nonnegative.");
        q_sum += q[i];
    }
    assert(p_sum==q_sum, "The sum of the row and column sums must agree.");
}


bigint* count(data_t* v, int matrix_type) {
    int m = v->m;
    int L = v->L;
    bigint* result;
    if (matrix_type==0) {
        result = count_recursion_binary(v->p,v->c,1,1,0,v->p[2],v->c[1],v);
    } else {
        int total = 0;
        for (int i = 1; i<=m; i++) total += v->p[i];
        result = count_recursion_natural(v->p,v->c,v->S,1,L-2,0,total,total,v);
    }
    int is_new;
    bigint* value = hash_table_put(v->table, v->c, &is_new);
    mpz_set(*value, *result);
    return result;
}

#ifndef MAIN
int main(int argc, char** argv) {
    if ((argc>4)||(argc<3)) {
        fprintf(stderr,"\nERROR: Incorrect usage. Aborting...\n\n");
        fprintf(stderr,"\nExample usage:\n\n");
        fprintf(stderr,"> %s input.dat table.bin 0\n\n",argv[0]);
        fprintf(stderr,"  1st argument = file containing matrix size, type, and margins, formatted as e.g.\n");
        fprintf(stderr,"                   13 17 0\n");
        fprintf(stderr,"                   14 13 14 10 12 2 10 1 10 11 6 2 17\n");
        fprintf(stderr,"                   4 4 11 10 10 8 9 10 8 9 3 10 4 7 9 3 3\n");
        fprintf(stderr,"  2nd argument = filename to use for saving binary data\n");
        fprintf(stderr,"  3rd argument = 0: normal output (default)\n");
        fprintf(stderr,"                 1: suppress all output except the number of matrices\n");
        fprintf(stderr,"\n\n");
        exit(1);
    }
    
    if (argc==4) sscanf(argv[3],"%i",&quiet_mode);
    else quiet_mode = 0; // default is normal output (quiet mode off)
    
    char* input_filename = argv[1];
    char* table_filename = argv[2];
    
    // Read p (row sums), q (column sums), and matrix_type from input file
    int m,n,matrix_type;
    FILE* file = fopen(input_filename, "r");
    fscanf(file,"%i %i %i",&m,&n,&matrix_type);
    int* p; allocate_vector(int,p,m);
    int* q; allocate_vector(int,q,n);
    for (int i = 0; i<m; i++) fscanf(file,"%i",&p[i]);
    for (int i = 0; i<n; i++) fscanf(file,"%i",&q[i]);
    fclose(file);
    
    // Echo what we read in
    print("\n");
    print("Input file: %s\n",input_filename);
    print("Table file: %s\n",table_filename);
    if (matrix_type==0) { print("Binary matrices\n"); }
    else if (matrix_type==1)  { print("Nonnegative integer matrices\n"); }
    else { error("Please use a matrix type of either 0 (binary) or 1 (nonnegative integer)."); }
    print("m = %d\n",m);
    print("n = %d\n",n);
    print("p = "); if (!quiet_mode) print_vector("%d ",p,m);
    print("q = "); if (!quiet_mode) print_vector("%d ",q,n);
    print("====================================\n");
    
    // Make sure p and q are valid.
    validate_margins(p,q,m,n);

    // Initialize counting variables
    data_t* v = initialize_variables(p,q,m,n,"");
    
    print("Counting...\n");
    
    // Start the clock (to time how long it takes to run)
    clock_t start_time = clock();
    
    // Count the number of matrices
    bigint* result = count(v,matrix_type);
                
    // Stop the clock
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time))/CLOCKS_PER_SEC;
    
    // Print results
    print("\n");
    print("Elapsed CPU time = %f seconds\n",elapsed_time);
    print("Number of matrices = ");
    gmp_printf("%Zd",*result); // in quiet mode, this is the only output
    
    // Save the hash table to file
    print("\n\n");
    print("Saving lookup table to file...\n");
    hash_table_save(v->table, table_filename);
    
    return 0;
}
#endif // MAIN











