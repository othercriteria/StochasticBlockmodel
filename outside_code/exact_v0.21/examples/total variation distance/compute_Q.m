function log_Q = compute_Q(p,q,X)
% Compute the (log) probability of the matrices X(:,:,i) under Q

[m,n,k] = size(X);
assert(m==length(p));
assert(n==length(q));

log_Q = zeros(k,1);
for i = 1:k
    [log_Q(i),~,~] = BernoulliMarginsRnd(1,p',q,[],[],[],[],X(:,:,i));
end







