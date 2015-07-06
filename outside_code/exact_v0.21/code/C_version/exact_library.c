#include "common.h"
#include "count_recursion.c"
#include "sample_recursion.c"

double count_miller(int m, int n, int *p, int *q) {
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

    // Initialize S and results with all zeros
    allocate_matrix(int,v->S,m+1,L);
    allocate_GMP_matrix(v->results,m+1,L);

    // Precompute all binomial coefficients that we might need
    precompute_binomial(n);

    // Initialize a new hash table
    v->table = hash_table_initialize(L,100000);

    // Put in the root entry
    int* root_key; allocate_vector(int,root_key,L); // all zeros
    int is_new;
    bigint* value = hash_table_put(v->table, root_key, &is_new);
    mpz_set_ui(*value, 1);

    bigint* result;
    result = count_recursion_binary(v->p,v->c,1,1,0,v->p[2],v->c[1],v);

    return mpz_get_d(*result);
}

double sample_miller(int **s, int m, int n, int *p, int *q) {
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

    // Initialize S and results with all zeros
    allocate_matrix(int,v->S,m+1,L);
    allocate_GMP_matrix(v->results,m+1,L);

    // Precompute all binomial coefficients that we might need
    precompute_binomial(n);

    // Initialize a new hash table
    v->table = hash_table_initialize(L,100000);

    // Put in the root entry
    int* root_key; allocate_vector(int,root_key,L); // all zeros
    int is_new;
    bigint* value = hash_table_put(v->table, root_key, &is_new);
    mpz_set_ui(*value, 1);

    // Fill hash table by enumerating
    count_recursion_binary(v->p,v->c,1,1,0,v->p[2],v->c[1],v);

    
}

// Uniformly sample a subset of size k from n elements
void sample_subset(int* x, int n, int k) {
    // initialize x to all zeros
    for (int i = 0; i<n; i++) x[i] = 0;

    // sample without replacement
    for (int i = 0; i<k; i++) {
        mpz_set_ui(max_number, n-i);
        mpz_urandomm(uniform_number, random_state, max_number);
        int v = mpz_get_ui(uniform_number);
        int j = 0;
        while (v>=0) {
            if (x[j]==0) v -= 1;
            j += 1;
        }
        x[j-1] = 1;
    }
}

void sample(int** matrix, data_t* v) {
    int m = v->m;
    int n = v->n;
    int L = v->L;
    // reset matrix to all zeros
    for (int i = 0; i<m; i++) {
        for (int j = 0; j<n; j++) matrix[i][j] = 0;
    }
    // reset S and results to all zeros
    for (int j = 0; j<m+1; j++) {
        for (int k = 0; k<L; k++) {
            v->S[j][k] = 0;
            mpz_set_ui(v->results[j][k], 0);
        }
    }

    bigint* number = hash_table_get(v->table,v->c);
    sample_recursion_binary(number,v->p,v->c,v->S,1,1,0,v->p[2],v->c[1],v);
    sample_matrix_entries(matrix,v);
}

void sample_matrix_entries(int** matrix, data_t* v) {
    int m = v->m;
    int n = v->n;
    int L = v->L;

    // make copies of q and c
    int* u; allocate_vector_copy(int,u,v->q,n);
    int* d; allocate_vector_copy(int,d,v->c,L);
    // allocate subset indicator vector
    int* x; allocate_vector(int,x,n);

    for (int j = 1; j<m+1; j++) {
      for (int k = 1; k<L-1; k++) {
	int r_k = d[k]-d[k+1];
	sample_subset(x, r_k, v->S[j][k]);
	int i = 0;
	int s = 0;
	while (s < r_k) {
	  if (u[i]==k) {
	    matrix[v->permutation[j-1]][i] = x[s];
	    u[i] -= x[s];
	    s += 1;
	  }
	  i += 1;
	}
	d[k] -= v->S[j][k];
      }
    }
    free(u);
    free(d);
    free(x);
}
