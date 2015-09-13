function X = sample_Q(p,q,k)
% Sample k matrices with margins (p,q) i.i.d. from Q.

m = length(p);
n = length(q);

% Draw k samples (in sparse form) i.i.d. from Q
[~,~,alist] = BernoulliMarginsRnd(k,p',q);

% Convert to non-sparse matrices
X = false(m,n,k);
for i = 1:k
    for t = 1:size(alist,2)
        X(alist(1,t,i),alist(2,t,i),i) = true;
    end
end

% Verify that the samples have the correct row and column sums
for i = 1:k
    assert(all(sum(X(:,:,i),1)==q));
    assert(all(sum(X(:,:,i),2)==p'));
end







