function A = MCMC(A,n_steps)
% Run the |1 0| <-> |0 1| swap MCMC algorithm on matrix A for n_steps.
%         |0 1|     |1 0|

[m,n] = size(A);

% Pre-generate random numbers.
random_m = randi(m, n_steps, 2);
random_n = randi(n, n_steps, 2);

for k = 1:n_steps
    % Randomly choose a 2x2 submatrix, swap |1 0| with |0 1|.
    %                                       |0 1|      |1 0|
    i_1 = random_m(k,1);
    i_2 = random_m(k,2);
    j_1 = random_n(k,1);
    j_2 = random_n(k,2);
    
    if (A(i_1,j_1)+A(i_1,j_2)==1) ...
    && (A(i_2,j_1)+A(i_2,j_2)==1) ...
    && (A(i_1,j_1)+A(i_2,j_1)==1) ...
    && (A(i_1,j_2)+A(i_2,j_2)==1)
    
        b = A(i_1,j_1);
        A(i_1,j_1) = 1-b;
        A(i_2,j_2) = 1-b;
        A(i_1,j_2) = b;
        A(i_2,j_1) = b;
    end
end
