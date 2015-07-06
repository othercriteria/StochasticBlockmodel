
#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <stdlib.h> 

// GMP (GNU Multiple Precision) library (for "bigint"s)
#include <gmp.h>
typedef mpz_t bigint;

#define max(a,b) (((a)>(b))?(a):(b))
#define min(a,b) (((a)<(b))?(a):(b))



#define safe_malloc(address,number,type) {                                                              \
    while ((address = malloc((number)*sizeof(type)))==NULL) {                                           \
        fprintf(stderr,"Memory allocation failed! Close some programs and hit enter to try to continue.\n");    \
        getchar();                                                                                      \
    }                                                                                                   \
}

#define safe_calloc(address,number,type) {                                                              \
    while ((address = calloc((number),sizeof(type)))==NULL) {                                           \
        fprintf(stderr,"Memory allocation failed! Close some programs and hit enter to try to continue.\n");    \
        getchar();                                                                                      \
    }                                                                                                   \
}

// Macro to allocate memory for an array and initialize to all zeros
// type = the type for the entries (e.g. int)
// type* x = pointer to the array
// int n = length of the array
#define allocate_vector(type,x,n)  safe_calloc(x,(n),type)

// Macro to allocate memory for a 2-D array and initialize to all zeros
// type = the type for the entries (e.g. int)
// type** x = pointer to the array
// int m, n = dimensions of the array
#define allocate_matrix(type,x,m,n) {                           \
    type* _data;                                                \
    safe_calloc(_data,(m)*(n),type);                       \
    safe_malloc(x,(m),type*);                              \
    for (int _i = 0; _i < (m); _i++) x[_i] = &_data[_i*(n)];    \
}

// Same as allocate_vector, but do not initialize to all zeros
#define mallocate_vector(type,x,n)  safe_malloc(x,(n),type)

// Same as allocate_matrix, but do not initialize to all zeros
#define mallocate_matrix(type,x,m,n) {                           \
    type* _data;                                                \
    safe_malloc(_data,(m)*(n),type);                       \
    safe_malloc(x,(m),type*);                              \
    for (int _i = 0; _i < (m); _i++) x[_i] = &_data[_i*(n)];    \
}

// Same as allocate_vector, but for GMP integers
#define allocate_GMP_vector(x,n) {                      \
    safe_malloc(x,(n),mpz_t);                           \
    for (int _i = 0; _i < (n); _i++) mpz_init(x[_i]);   \
}

// Same as allocate_matrix, but for GMP integers
#define allocate_GMP_matrix(x,m,n) {                           \
    mpz_t* _data;                                                \
    safe_malloc(_data,(m)*(n),mpz_t);                       \
    safe_malloc(x,(m),mpz_t*);                              \
    for (int _i = 0; _i < (m); _i++) x[_i] = &_data[_i*(n)];    \
    for (int _i = 0; _i < (m); _i++) {                          \
        for (int _j = 0; _j < (n); _j++) mpz_init(x[_i][_j]);   \
    }                                                           \
}
    
#define allocate_vector_copy(type,x,y,n) {      \
    allocate_vector(type,x,n);                  \
    for (int _i = 0; _i<n; _i++) x[_i] = y[_i]; \
}
    
#define allocate_matrix_copy(type,x,y,m,n) {                    \
    allocate_matrix(type,x,m,n);                                \
    for (int _i = 0; _i<(m); _i++) {                            \
        for (int _j = 0; _j<(n); _j++) x[_i][_j] = y[_i][_j];   \
    }                                                           \
}

// I'm using a macro for these so that they will be generic for the type of x
#define print_vector(format_string,x,n) {                         \
    for (int _i = 0; _i<(n); _i++) printf(format_string, x[_i]);  \
    printf("\n");                                                 \
}

#define print_matrix(format_string,x,m,n) {         \
    for (int _i = 0; _i<(m); _i++) {                \
        for (int _j = 0; _j<(n); _j++)              \
            printf(format_string, x[_i][_j]);       \
        printf("\n");                               \
    } printf("\n");                                 \
}

#define print_GMP_vector(format_string,x,n) {                         \
    for (int _i = 0; _i<(n); _i++) gmp_printf(format_string, x[_i]);  \
    printf("\n");                                                     \
}
    
#define print_GMP_matrix(format_string,x,m,n) {          \
    for (int _i = 0; _i<(m); _i++) {                \
        for (int _j = 0; _j<(n); _j++)              \
            gmp_printf(format_string, x[_i][_j]);       \
        printf("\n");                               \
    } printf("\n");                                 \
}

// Print error message and exit the program.
void error(char* message) {
    fprintf(stderr,"\nERROR: "); 
    fprintf(stderr,"\n");
    exit(1);
}

// Customized assert
static inline void assert(int condition, char* message) {
    if (!condition) error(message);
}

#ifdef ERROR_CHECKING
#define assert_if_enabled(condition,message) assert(condition,message)
#else
#define assert_if_enabled(condition,message)  ;
#endif



// Sort x in descending order, and return the permutation used.
int* sort(int* x, int n) {
    int* permutation;
    allocate_vector(int,permutation,n);
    for (int i = 0; i<n; i++) permutation[i] = i;
    
    for (int i=0; i<n; i++) {
        for (int j=0; j<n-1; j++) {
            if (x[j] < x[j+1]) {
                int value = x[j+1];
                x[j+1] = x[j];
                x[j] = value;
                
                value = permutation[j+1];
                permutation[j+1] = permutation[j];
                permutation[j] = value;
            }
        }
    }
    return permutation;
}
    
#endif // UTILITIES_H


