% This is a simple check on the uniformity of the sampler.
% 
% It draws a number of samples, and compares the empirical distribution to the uniform distribution. The ratios P_empirical/P_uniform should all approach 1 as the number of samples goes to infinity.

fprintf('====== Uniformity of the samples ======\n');

p = [3 1 2]
q = [1 2 1 2]
k = 10000; % number of samples

for matrix_type = [0 1]
    switch matrix_type
        case 0, fprintf('BINARY CASE:\n');
        case 1, fprintf('NONNEGATIVE INTEGER CASE:\n');
    end
    
    % Count and sample
    number = count(p,q,matrix_type);
    fprintf('Number of matrices = %s\n',number);
    fprintf('Sampling %d matrices...\n',k);
    X = sample(p,q,k);

    % Compute histogram of sampled matrices
    histogram = containers.Map();
    for i = 1:k
        key = mat2str(X(:,:,i));
        if histogram.isKey(key)
            histogram(key) = histogram(key) + 1;
        else
            histogram(key) = 1;
        end
    end
    fprintf('Histogram of sampled matrices:\n');
    fprintf('(index)  (count)  (P_empirical/P_uniform)\n');
    values = histogram.values();
    for i = 1:length(values)
        fprintf('%d %d %f\n',i,values{i}, str2num(number)*values{i}/k);
    end
    fprintf('\n\n');
end

    
    