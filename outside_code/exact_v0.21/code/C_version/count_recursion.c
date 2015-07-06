#ifndef COUNT_RECURSION
#define COUNT_RECURSION

// The main recursion (binary mode)
bigint* count_recursion_binary(int* p, int* c, int j, int k, int s_sum, int p_sum, int c_sum,
                               data_t* v)
{
    bigint* result;
    
    if (s_sum == p[j]) {  // lookup the value or compute it if unknown
        // try to lookup c in the hash table
        result = hash_table_get(v->table,c);
        if (!result) {  // if it's not there, recurse
            result = count_recursion_binary(p,c,j+1,1,0,p[j+2],c[1],v);
            
            // if ((c[1] != p[j+1]) && (j%10==0)) {
            if (c[1] != p[j+1]) {
                int is_new;
                bigint* value = hash_table_put(v->table, c, &is_new);
                mpz_set(*value, *result);
            }
        } 
    } else {
        int r_k = c[k] - c[k+1];
        
        // Upper bound on s_k according to Gale-Ryser
        int GR_condition = c_sum - s_sum - p_sum;
        // range of values to sum over for s_k
        int smallest = max(0, p[j] - s_sum - c[k+1]);
        int largest = min(min(r_k, p[j] - s_sum), GR_condition);

        result = &(v->results[j][k]);
        mpz_set_ui(*result, 0);
        for (int s_k = smallest; s_k<=largest; s_k++) {
            c[k] -= s_k;
            bigint* value = count_recursion_binary(p,c,
                j,k+1, s_sum+s_k, p_sum + p[j+k+1], c_sum + c[k+1], v);
            c[k] += s_k;
            mpz_addmul(*result, binomial[r_k][s_k], *value);
        }
        // print("result = %d\n",result);
    }
    return result;
}

// The main recursion (nonnegative integer mode)
bigint* count_recursion_natural(int* p, int* c, int** S, int j, int k, int s_sum, int r_sum, int total,
                                data_t* v)
{
    bigint* result;
    
    if (s_sum == p[j]) {  // lookup the value or compute it if unknown
        // try to lookup c in the hash table
        for (int i = 1; i < v->L; i++) c[i] -= S[j][i];
        result = hash_table_get(v->table,c);
        if (!result) {  // if it's not there, recurse
            result = count_recursion_natural(p,c,S,j+1,(v->L)-2,0,total-p[j], total-p[j], v);
            
            int is_new;
            bigint* value = hash_table_put(v->table, c, &is_new);
            mpz_set(*value, *result);
        }
        for (int i = 1; i < v->L; i++) c[i] += S[j][i];
    } else {
        //while (c[k]-c[k+1] + S[j][k+1] == 0)  k-=1;
        
        int r_k = c[k] - c[k+1];
        r_sum -= k*r_k;
        
        // range of values to sum over for s_k
        int smallest = ceil(max(0, (p[j] - s_sum - r_sum)/((double)k)));
        int largest = min(r_k + S[j][k+1], p[j] - s_sum);

        result = &(v->results[j][k]);
        mpz_set_ui(*result, 0);
        for (int s_k = smallest; s_k<=largest; s_k++) {
            S[j][k] = s_k;
            bigint* value = count_recursion_natural(p,c,S,j,k-1, s_sum+s_k, r_sum,total,v);
            mpz_addmul(*result, binomial[r_k + S[j][k+1]][s_k], *value);
        }
        // print("result = %d\n",result);
        S[j][k] = 0;
    }
    return result;
}


#endif // COUNT_RECURSION










