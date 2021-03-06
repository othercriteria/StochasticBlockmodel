function A = generate_matrix(a,b)
% Return a binary matrix with column sums a and row sums b.
%
% Inputs:
%   a (n x 1) nonincreasing (integer) vector of column sums
%   b (m x 1) (integer) vector of row sums
% Output:
%   A (m x n) binary matrix with column sums a and row sums b
%
% If there is no such matrix, an error is produced.
%
% Code provided by Jeff Miller (jeffrey_miller@brown.edu)

n = size(a,1);
m = size(b,1);

% Make sure everything is proper.
assert(all(a>=0)) % nonnegative
assert(all(b>=0))
assert(all(mod(a,1)==0)) % integer
assert(all(mod(b,1)==0))
assert(all(diff(a)<=0)) % a nonincreasing

% Construct the maximal matrix and the conjugate.
A = zeros(m,n);
for i = 1:m
    A(i,1:b(i)) = 1;
end
c = sum(A,1)';

% Check whether a matrix exists (Gale-Ryser conditions).
assert(sum(a)==sum(c));
assert(all(cumsum(a)<=cumsum(c)));

% Convert the maximal matrix into one with column sums a.
% (This procedure is guaranteed to terminate at a correct matrix.)
while any(c~=a)
    j = find(c>a, 1);
    k = find(c<a, 1);
    i = find(A(:,j) > A(:,k), 1);
    A(i,j) = 0;
    A(i,k) = 1;
    c(j) = c(j)-1;
    c(k) = c(k)+1;
end

% Verify that it worked.
assert(all(a==sum(A,1)'))
assert(all(b==sum(A,2)))
