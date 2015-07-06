#include "count_recursion.c"

// The main recursion (binary version)
void sample_recursion_binary(bigint* number, 
                             int* p, int* c, int** S, int j, int k, int s_sum, int p_sum, int c_sum,
                             data_t* v)
{
    bigint* result;
    
    if (s_sum==p[j]) {  // proceed to the next row
    
        if (j == (v->m)+1) return; // if we are done, return
        
        else sample_recursion_binary(number,p,c,S,j+1,1,0,p[j+2],c[1],v); // otherwise, recurse
        
    } else {
        int r_k = c[k] - c[k+1];
        
        // Upper bound on s_k according to Gale-Ryser
        int GR_condition = c_sum - s_sum - p_sum;

        // range of values to sum over for s_k
        int smallest = max(0, p[j] - s_sum - c[k+1]);
        int largest = min(min(r_k, p[j] - s_sum), GR_condition);

        // Sample uniformly from {0,1,...,number-1}
        // (Make sure to do this before setting results[j][k] to 0.)
        mpz_urandomm(uniform_number, random_state, *number);
        
        // Find the value of s_k to use
        result = &(v->results[j][k]);
        mpz_set_ui(*result, 0); // set result to 0
        for (int s_k = smallest; s_k<=largest; s_k++) {
            c[k] -= s_k;
            bigint* value = count_recursion_binary(p,c,j,k+1, s_sum+s_k, p_sum+p[j+k+1],c_sum+c[k+1],v);
            mpz_addmul(*result, binomial[r_k][s_k], *value);
            
            if (mpz_cmp(*result,uniform_number) > 0) { // if result>uniform_number
                S[j][k] = s_k;
                sample_recursion_binary(value,p,c,S,j,k+1, s_sum+s_k, p_sum+p[j+k+1], c_sum+c[k+1], v);
                c[k] += s_k;
                return;
            }
            c[k] += s_k;
        }
        error("Internal error: This should be unreachable!");
    }
    
    return;
}


// The main recursion (nonnegative integer version)
void sample_recursion_natural(bigint* number, 
                              int* p, int* c, int** S, int j, int k, int s_sum, int r_sum, int total,
                              data_t* v)
{                         
    bigint* result;
    
    if (s_sum==p[j]) {  // proceed to the next row
    
        if (j == (v->m)+1) {
            return; // if we are done, return
        } else { // otherwise, recurse
            for (int i = 1; i < v->L; i++) c[i] -= S[j][i];
           
            sample_recursion_natural(number,p,c,S,j+1,(v->L)-2,0,total-p[j],total-p[j],v);
            
            for (int i = 1; i < v->L; i++) c[i] += S[j][i];
        }
    } else {
        // while (c[k]-c[k+1] + S[j][k+1] == 0)  k-=1;
        
        int r_k = c[k] - c[k+1];
        r_sum -= k*r_k;
        
        // range of values to sum over for s_k
        int smallest = ceil(max(0, (p[j] - s_sum - r_sum)/((double)k)));
        int largest = min(r_k + S[j][k+1], p[j] - s_sum);

        // Sample uniformly from {0,1,...,number-1}
        // (Make sure to do this before setting results[j][k] to 0.)
        mpz_urandomm(uniform_number, random_state, *number);
        
        // Find the value of s_k to use
        result = &(v->results[j][k]);
        mpz_set_ui(*result, 0); // set result to 0
        for (int s_k = smallest; s_k<=largest; s_k++) {
            S[j][k] = s_k;
            bigint* value = count_recursion_natural(p,c,S, j,k-1, s_sum+s_k, r_sum, total, v);
            mpz_addmul(*result, binomial[r_k + S[j][k+1]][s_k], *value);
            
            if (mpz_cmp(*result,uniform_number) > 0) { // if result>uniform_number
                sample_recursion_natural(value,p,c,S, j,k-1, s_sum+s_k, r_sum, total, v);
                return;
            }
        }
        error("Internal error: This should be unreachable!");
    }
    
    return;
}

