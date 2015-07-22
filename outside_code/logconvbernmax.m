function p = logconvbernmax(b,nmax)
% function logp = logconvbernmax(b,nmax=n)
%
% b is a vector of n probabilities
%
% p is a (nmax+1)-vector with p(j)=log(Prob(X=j-1))
% for a random variable X that is the sum of
% n independent Bernoulli(b(k))'s
%
% Values for b outside of [0,1] are forced to {0,1}.
%
% p is a column vector

% Matt Harrison
% June 19, 2015

m = numel(b);

if nargin < 2 || isempty(nmax), nmax = m; end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREPROCESSING                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% remove ones and zeros
zerocount = 0;
onecount = 0;

j = 0;
for k = 1:m
	bk = b(k);
	if j
		b(k-j) = bk;
	end
	if bk <= 0
		zerocount = zerocount + 1;
		j = j + 1;
	elseif bk >= 1
		onecount = onecount + 1;
		j = j + 1;
	end
end
m = m-j;
n = nmax - onecount;

% check for trivial
if n < 0
    p = -inf(nmax+1,1);
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN ALGORITHM                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = min(n,m)+1;
p = zeros(n1,1);

% convolve each bernoulli
for k = 1:m
	
	% precomputations
	bk = b(k);
	logbk = log(bk);
	logbk1 = log(1-bk);
	
	% compute the probability of k (all ones) in the first k
	if k < n1, p(k+1) = p(k)+logbk; end
	
	% loop backwards 
	for j = min(k,n1):-1:2
		% current value is mixture of current and previous
		x = p(j-1)+logbk;
		y = p(j)+logbk1;
		% be careful to avoid underflow when computing
		% p(j) = log(exp(x)+exp(y));
		if x < y
			p(j) = y + log(1+exp(x-y));
		else
			p(j) = x + log(1+exp(y-x));
		end
	end
	
	% compute the probability of 0 (all zeros) in the first k
	p(1) = p(1)+logbk1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POSTPROCESSING                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% account for zeros and ones
p = [-inf(onecount,1); p ; -inf(nmax+1-onecount-n1,1)];
