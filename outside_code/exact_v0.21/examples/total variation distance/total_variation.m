% Estimate the total variation distance between the uniform distribution P and an approximation Q.
% 
% This can be used on any approximation Q for which it is possible to sample and compute probabilities.
% Samples from Q are obtained using sample_Q, and the probability of a sample is found with compute_Q.


number_of_samples = 10000;
confidence = .999;


examples_to_run = [1.1,2.2, [4.1:.1:4.5]];

for example = examples_to_run
    switch example
        % ==================================================================
        % Ecology co-occurrence matrices
        case 1.1
        title = 'Darwin''s finches';
        p = [14 13 14 10 12 2 10 1 10 11 6 2 17];
        q = [4 4 11 10 10 8 9 10 8 9 3 10 4 7 9 3 3];

        case 1.2
        title = 'Manly''s lizards';
        p = [7 23 6 6 2 18 8 8 1 22 2 2 9 2 1 18 9 3 3 1];
        q = [13 4 9 4 3 2 3 4 4 5 2 5 2 10 10 10 7 6 6 3 3 11 8 11 6];
        
        case 1.3
        title = 'Patterson & Atmar''s montane mammals';
        p = [24 23 21 19 13 13 12 11 10 10 9 9 7 7 7 7 7 7 6 6 5 5 4 3 2 1 1]; 
        q = [25 25 24 21 21 17 11 11 11 10 9 9 7 7 7 6 5 5 4 4 3 3 2 2];
        
        % ==================================================================
        % Social networks
        case 2.1
        title = 'Sampson''s monks';
        p = [9 7 2 3 5 1 4 4 2 1 2 1 2 6 1 2 1 2]; 
        q = [3 3 3 3 3 3 3 3 3 3 3 3 3 4 3 3 3 3];
        
        case 2.2
        title = 'MacRae''s prisoners';
        p = [0 1 1 4 4 2 4 5 3 1 1 4 2 1 3 7 3 3 1 3 6 2 2 3 2 2 0 4 2 5 0 1 4 3 1 0 6 1 1 2 5 2 2 1 3 4 6 6 4 1 4 8 0 3 7 7 4 2 1 1 1 1 0 4 0 1 4];
        q = [2 3 3 1 3 1 1 8 3 2 1 4 2 2 3 2 2 4 0 3 2 3 3 3 0 0 3 3 1 5 4 3 2 2 0 3 6 1 4 2 5 2 2 3 4 3 2 2 4 3 1 3 3 5 4 4 1 3 2 2 4 5 5 3 2 3 2];

        % ==================================================================
        % Regular matrices
        otherwise
        m = 30; n = m; r = round((example-4)*10);
        title = sprintf('%d x %d matrices with %d-regular margins',m,n,r);
        p = r*ones(1,m);
        q = r*ones(1,n);;
    end
    
    % Summary
    fprintf('\n============== %s ==============\n',title);
    fprintf('p = [ '); fprintf('%d ',p); fprintf(']\n');
    fprintf('q = [ '); fprintf('%d ',q); fprintf(']\n');

    % Count the number of binary matrices with row sums p and column sums q
    fprintf('Counting...\n');
    number = count(p,q);
    fprintf('Number of matrices = %s\n',number);

    % Draw i.i.d. samples from P and Q
    fprintf('Sampling %d matrices from P...\n',number_of_samples);
    U = sample(p,q,number_of_samples);
    fprintf('Sampling %d matrices from Q...\n',number_of_samples);
    V = sample_Q(p,q,number_of_samples);


    % Convert these to i.i.d. samples from (P + Q)/2
    fprintf('Sampling %d Bernoulli(1/2) variables...\n',number_of_samples);
    Z = (rand(number_of_samples,1)<.5);
    X = zeros(size(U));
    X(:,:,Z) = U(:,:,Z);
    X(:,:,~Z) = V(:,:,~Z);

    % Compute the Q probability of each sample. (We know the P probability is 1/number.)
    fprintf('Computing Q probabilities...\n');
    log_Q = compute_Q(p,q,X);

    % Compute the likelihood ratios
    log_number = log(str2num(number)); % (this is approximate, which is inevitable for this calculation)
    ratios = exp(log_Q + log_number); % Q(x)/P(x)

    % Compute the estimated total variation distance
    values = abs((1-ratios)./(1+ratios));
    TV_estimate = mean(values);
    standard_deviation = std(values) / sqrt(number_of_samples);

    fprintf('Estimated total variation distance = %.12f\n',TV_estimate);
    fprintf('Estimated standard deviation of the estimator = %.12f\n',standard_deviation);

    % Compute a confidence interval for the estimate (using Hoeffding's inequality)
    alpha = 1-confidence;
    epsilon = sqrt(-log(alpha/2)/(2*number_of_samples));
    fprintf('Confidence interval:\n');
    fprintf(' With probability %f, this estimator will be within %.12f of the true value.\n',confidence,epsilon);

end














