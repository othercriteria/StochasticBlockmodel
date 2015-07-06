
#include "khash.h"
#include "utilities.h"  
#include "lookup3.c"


// Generate hash table declarations
// This defines a hash table with integer array keys.
// The 1st entry key[0] must specify the length of the key (not including entry key[0]).
int vector_equal(int* x, int* y, int n) {
    for (int i = 0; i<n; i++) if (x[i] != y[i]) return 0;
    return 1;
}
#define int_array_hash_fn(x) hashword((uint32_t*)&x[1], (size_t)x[0], 0) 
#define int_array_hash_eq(x,y) (khint_t)vector_equal(x,y,x[0]+1)
KHASH_INIT(H, int*, bigint*, 1, int_array_hash_fn, int_array_hash_eq)

// uint32_t hashword(
// const uint32_t *k,                   /* the key, an array of uint32_t values */
// size_t          length,               /* the length of the key, in uint32_ts */
// uint32_t        initval)         /* the previous hash, or an arbitrary value */

// Table structure type
typedef struct {
    khash_t(H)* h; // khash hash table
    int** keys; // current block of key strings
    bigint* values; // current block of values
    int index; // memory index for (keys,values)
    int L; // length of each key
    int block_size; // number of entries in each memory block of (key,value) pairs
} table_t;


// Initialize a hash table.
// L = length of int arrays (with a 0 at the 1st entry) to use for keys.
table_t* hash_table_initialize(int L, int block_size) {
    // Allocate memory for the struct
    table_t* t = malloc(sizeof(*t));

    // Allocate initial block of memory for keys and values
    mallocate_matrix(int,t->keys,block_size,L);
    allocate_GMP_vector(t->values,block_size);
    
    t->index = 0;
    t->L = L;
    t->block_size = block_size;
    
    // Initialize the khash table
    t->h = kh_init(H);
    
    return t;
}

// Lookup the given key in the table, and return the address of the associated value.
// If the key is not in the table, return 0.
bigint* hash_table_get(table_t* t, int* key) {  
    
    // Since we are using 1-indexing for the integer array, we use the 1st entry to indicate length.
    key[0] = (t->L)-1;
    
    // Try to look it up
    khiter_t position = kh_get(H, t->h, key);
    if (position == kh_end(t->h)) return 0;
    else return kh_value(t->h,position);
}

// Enter the given key into the table and return the address in which to store the associated value.
// If the given key is already in the table, return the address of the associated value.
bigint* hash_table_put(table_t* t, int* key, int* _is_new) {  
    bigint* value_address;
    
    // Copy key to memory, since khash will store only the address.
    int* next_key = t->keys[t->index];
    for (int i = 1; i < t->L; i++)  next_key[i] = key[i];
    // Since we are using 1-indexing for the integer array, we use the 1st entry to indicate length.
    next_key[0] = (t->L)-1;
    
    // Put the key in the hash table
    khiter_t position = kh_put(H, t->h, next_key, _is_new);
    
    // If the key is new, get its corresponding value address, and update the memory index.
    if (*_is_new) { 
        value_address = &(t->values[t->index]);
        kh_value(t->h,position) = value_address;
        t->index += 1;
        
        // If we've used up all the keys in the currently allocated block, allocate a new block.
        // (Make keys of length L in order to leave a 0 at the end, just in case c has no zeros.)
        // (Do NOT free the memory of previous keys and values --- we need them for later lookups.)
        if (t->index==t->block_size) {
            mallocate_matrix(int,t->keys,t->block_size,t->L);
            allocate_GMP_vector(t->values,t->block_size);
            t->index = 0;
        }
    } else {
        value_address = kh_value(t->h,position);
    }
    
    return value_address;
}

// Save the hash table to file.
void hash_table_save(table_t* t, char* filename) {
    // open the output file
    FILE* file;
    file = fopen(filename, "wb");
    
    // write header data
    khint_t total = kh_size(t->h);
    fwrite(&total, sizeof(total), 1, file); // total number of entries
    fwrite(&(t->L), sizeof(t->L), 1, file); // length of each key
    
    // write data for each (key,value) pair in the table
    for (khiter_t i = kh_begin(t->h); i != kh_end(t->h); i++) {
        if (kh_exist(t->h, i)) {
            size_t n_words;
            unsigned long *x;
            bigint* value = kh_value(t->h, i);
            x = mpz_export(NULL, &n_words, 1, sizeof(*x), 1, 0, *value);
            fwrite(kh_key(t->h,i), sizeof(int), t->L, file);
            fwrite(&n_words, sizeof(n_words), 1, file);
            fwrite(x, sizeof(*x), n_words, file);
        }
    }
    fclose(file);
}


// Load hash table from file. (Includes initialization of the hash table.)
table_t* hash_table_load(char* filename, int block_size) {
    // open the input file
    FILE* file;
    file = fopen(filename, "rb");
    
    // read header data
    // size_t n_read;
    khint_t total;
    int L;
    fread(&total, sizeof(total), 1, file); // total number of entries
    fread(&L, sizeof(L), 1, file); // length of each key
    
    // initialize table
    table_t* t = hash_table_initialize(L,block_size);
    
    // initialize buffers
    unsigned long *x;
    allocate_vector(unsigned long,x,100);
    int x_length = 100;
    int* key;
    allocate_vector(int,key,L);
    
    // read data for each (key,value) pair and put it in the table
    for (int i = 0; i<total; i++) {
        size_t n_words;
        int is_new;
        
        // read and store the key
        fread(key, sizeof(*key), L, file);
        bigint* value = hash_table_put(t, key, &is_new);
        
        //read and store the value
        fread(&n_words, sizeof(n_words), 1, file);
        if (n_words>x_length) {
            free(x);
            allocate_vector(unsigned long,x,2*n_words);
            x_length = 2*n_words;
        }
        fread(x, sizeof(*x), n_words, file);
        mpz_import(*value, n_words, 1, sizeof(*x), 1, 0, x);
    }
    fclose(file);
    return t;
}














