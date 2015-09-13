% Use MCMC to estimate the p-value for Darwin's finches dataset
% using Roberts & Stone (1990)'s $\bar S^2$ statistic.
fprintf('\n============== Null model analysis (MCMC): Darwin''s finches ==============\n');

% number of samples to use
n_samples = 10000;

% Darwin's finches
% 13 species (rows) and 17 islands (columns)
% Note: For the statistic below, we assume (rows,columns) correspond to (species,islands) respectively.
matrix_type = 0; % 0:binary, 1:nonnegative integer
A = [0 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 0
     1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 0 0
     0 0 1 1 1 0 0 1 0 1 0 1 1 0 1 1 1
     1 1 1 0 1 1 1 1 1 1 0 1 0 1 1 0 0
     0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0
     0 0 1 1 1 1 1 1 1 0 0 1 0 1 1 0 0
     0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
     0 0 1 1 1 1 1 1 1 1 0 1 0 0 1 0 0
     0 0 1 1 1 1 1 1 1 1 0 1 0 1 1 0 0
     0 0 1 1 1 0 1 1 0 1 0 0 0 0 0 0 0
     0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1];
p = sum(A,2)';
q = sum(A,1);
assert(all(p==[14 13 14 10 12 2 10 1 10 11 6 2 17]));
assert(all(q==[4 4 11 10 10 8 9 10 8 9 3 10 4 7 9 3 3]));

[m,n] = size(A);

% Compute the test statistic on the input data
C = A*A';
S = sum(sum(triu(C.^2)))/nchoosek(m,2);

% Draw i.i.d. samples using an MCMC approximation to the uniform distribution
% and compute the test statistic on the samples.
fprintf('Sampling %d matrices...\n',n_samples);
start_time = cputime;
random_m = randi(m,n_samples,2);
random_n = randi(n,n_samples,2);
statistics = zeros(n_samples,1);
m_choose_2 = nchoosek(m,2);
for k = 1:n_samples
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
    
    C = A*A';
    statistics(k) = sum(sum(triu(C.^2)))/m_choose_2;
end
elapsed_time = cputime-start_time;

assert(all(p==sum(A,2)'));
assert(all(q==sum(A,1)));

% Estimate the p-value
p_value = mean(statistics>=S);

fprintf('Estimated p-value = %.12f\n',p_value);
fprintf('Elapsed CPU time = %.12f seconds\n',elapsed_time);
fprintf('Time per sample = %.12f seconds\n',elapsed_time/n_samples);








