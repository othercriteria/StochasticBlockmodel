#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h> 
#include <string.h> 
#include <time.h>
#include <math.h>
 

// Memory allocation macros and various other utilities
#define ERROR_CHECKING
#include "utilities.h"

// Wrapper for khash with GMP values
#include "hash_table.h"

// Enable/disable printf statements dynamically
int quiet_mode = 0;
#define print  if(!quiet_mode) printf

// Binomial coefficients
bigint** binomial;

// Random number generator stuff
gmp_randstate_t random_state;
bigint uniform_number, max_number;

// Counting variable structure
typedef struct {
    int *p, *q, *c, *permutation; 
    int m,n,L;
    int** S;
    bigint** results;
    table_t* table;
} data_t;



// Compute the conjugate partition c of q.
// n = length of q
// L = length to use for c
int* conjugate(int* q, int n, int L) {
    int* c; allocate_vector(int,c,L);
    for (int i = 1; i<L; i++) {
        for (int j = 0; j<n; j++)
            c[i] += (q[j] >= i);
    }
    return c;
}


// Precompute all the binomial coefficients we might need
void precompute_binomial(int max_n) {
    // Initialize and set to 0
    allocate_GMP_matrix(binomial,max_n+1,max_n+1);
    
    // Compute Pascal's triangle
    mpz_set_ui(binomial[0][0], 1);
    for (int n = 1; n <= max_n; n++) {
        mpz_set_ui(binomial[n][0], 1);
        for (int k = 1; k <= n; k++)
            mpz_add(binomial[n][k], binomial[n-1][k], binomial[n-1][k-1]);
    }
}

// Initialize
data_t* initialize_variables(int* p, int* q, int m, int n, char* table_filename) {
    // Allocate memory for the data struct
    data_t* v = malloc(sizeof(*v));
    v->m = m;
    v->n = n;
    
    // Find max(q) and set L (length to use for c)
    int max_q = 0;
    for (int i = 0; i<n; i++) max_q = max(max_q,q[i]);
    int L = max_q+2;
    v->L = L;
    
    // Compute the conjugate of q
    v->c = conjugate(q,n,L);
    // print("c = "); print_vector("%d ",c,L);
    
    // Copy q into v
    allocate_vector_copy(int,v->q,q,n);
    
    // Initialize p
    // Make a copy, so we don't modify the input argument
    int* p_copy; allocate_vector_copy(int,p_copy,p,m);
    // Sort in p descending order, and get the permutation used
    v->permutation = sort(p_copy,m);
    
    // Pad p with zeros (including a 0 in front to make it like Matlab indexing)
    allocate_vector(int,v->p,1+m+L);
    for (int i = 0; i<m; i++) v->p[i+1] = p_copy[i];
    // print("p = "); print_vector("%d ",p,1+m+L);

    // Initialize S and results with all zeros
    allocate_matrix(int,v->S,m+1,L);
    allocate_GMP_matrix(v->results,m+1,L);
    // print("S =\n"); print_matrix("%d ",v->S,m+1,L);
    // print("results =\n"); print_GMP_matrix("%Zd ",v->results,m+1,L);
    
    // Precompute all binomial coefficients that we might need
    precompute_binomial(n);
    // print("binomial coefficients =\n"); print_GMP_matrix("%Zd ",binomial,n+1,n+1);
    
    if (strlen(table_filename)==0) { // If no table filename is given
        // Initialize a new hash table
        v->table = hash_table_initialize(L,100000);
        
        // Put in the root entry
        int* root_key; allocate_vector(int,root_key,L); // all zeros
        int is_new;
        bigint* value = hash_table_put(v->table, root_key, &is_new);
        mpz_set_ui(*value, 1);
        // print("root key = "); print_vector("%d ",key,L);
        
    } else {
        // Initialize and load hash table from file
        print("Loading hash table from file...\n");
        v->table = hash_table_load(table_filename,100000);
    }
    
    // Initialize random number stuff
    mpz_init(uniform_number);
    mpz_init(max_number);
    gmp_randinit_default(random_state);
    gmp_randseed_ui(random_state,(unsigned long)time(0));
    
    return v;
}

#endif // COMMON_H
