function [logQ,logP,alist] = BernoulliMarginsRnd(SampN,rN,cN,wN,pflag,wflag,cflag,bIN)
%function [logQ,logP,alist] = BernoulliMarginsRnd(N,r,c,w,pflag,wflag,cflag,Binput)
%
% Approximate sampling from independent Bernoulli random variables B(i,j)
% arranged as an m x n matrix B given the m-vector of row sums r and the
% n-vector of column sums c, i.e., given that sum(B,2)=r and sum(B,1)=c.
%
% An error is generated if no binary matrix agrees with r and c.
%
% B(i,j) is Bernoulli(p(i,j)) where p(i,j)=w(i,j)/(1+w(i,j)), i.e.,
% w(i,j)=p(i,j)/(1-p(i,j)).  [The case p(i,j)=1 must be handled by the user
% in a preprocessing step, by converting to p(i,j)=0 and decrementing the
% row and column sums appropriately.]  
%
% Use w=[] for w identically 1, i.e., approximate uniform sampling over
% binary matrices with margins r and c.
%
% N is the sample size.  Because of pre-processing, it is more efficient
% per matrix to use larger sample sizes.
% 
% alist stores the locations of the ones in the samples.  
% If d = sum(r) = sum(c), then alist is 2 x d x N.
%
% The 1-entries of the kth matrix are stored as alist(:,:,k).  The
% (row,column) indices are (alist(1,t,k),alist(2,t,k)) for t=1:d.
% 
% If B is the kth matrix, then B can be created from alist via:
%
% B = false(m,n); for t = 1:size(alist,2), B(alist(1,t,k),alist(2,t,k)) = true; end
%
% logQ(k)=log(probability that algorithm generates B)
% logP(k)=log(prod(w(B)))
%
% If the algorithm is used for importance sampling, then the kth
% unnormalized importance weight is exp(logP(k)-logQ(k)).
%
% NOTE for w(i,j)=0:
%
% If the entries of w are not strictly positive, then the algorithm can 
% sometimes generate matrices with logP(k)=-inf.  In these cases, some of
% the entries of alist(:,:,k) may be zero and logQ(k) corresponds to the
% probability of generating that particular alist(:,:,k).
%
% OPTIONS:
%
% pflag: 'canfield' or '' (default, works best in most cases)
%        'greenhill' (perhaps useful for sparse and highly irregular margins)
% pflag controls which combinatorial approximations are used
%
% wflag: 'sinkhorn' or '' (default)
% wflag controls the initial balancing of w; it is passed to canonical.m
%
% cflag: 'descend' or '' (default)
%        'none' (sample columns in original order)
% cflag controls the order in which the columns are sampled
%
% Binput is a m x n binary matrix.  If it is provided, then the algorithm
% computes the probability of generating this matrix.
%
%
% Copyright (c) 2012, Matthew T. Harrison
% All rights reserved.
%
% This implements the approximate sampler described in Harrison and Miller (2012),
% "Importance sampling for weighted binary matrices with specified margins" (submitted).
% http://www.dam.brown.edu/people/harrison/
%

if nargin < 8 || isempty(bIN)
    doIN = false;
else
    doIN = true;
end
if nargin < 7 || isempty(cflag)
    cflag = 'descend';
end
if nargin < 6 || isempty(wflag)
    wflag = 'sinkhorn';
end
if nargin < 5 || isempty(pflag)
    pflag = 'canfield';
end
if nargin < 4
    wN = [];
end

doW = true;
if isempty(wN), doW = false; end

doA = true;
if nargout < 2, doA = false; end

if ~isscalar(SampN) || SampN < 1 || SampN ~= round(SampN), error('SampN must be a positive integer'), end

ptype = 0;
switch lower(pflag)
    case 'canfield'
        ptype = 1;
    case 'greenhill'
        ptype = 2;
    otherwise
        error('unknown pflag')
end

%------------------------------------------------------%
%--------------- START: PREPROCESSING -----------------%
%------------------------------------------------------%

% sizing
mT = numel(rN);
nT = numel(cN);

% sort the marginals (descending)
rT = rN(:);
[rsort,rndxT] = sort(rT,'descend');

if doW
    % balance the weights
    [~,~,wopt] = canonical(wN,wflag);
    % reorder the columns
    switch lower(cflag)
        case 'none'
            cndx = 1:nT;
        case 'descend'
            [~,cndx] = sortrows(-[cN(:) var(wopt,0,1).']);
        otherwise
            error('unknown cflag')
    end
    csort = cN(cndx);
    wopt = wopt(:,cndx);
    % precompute log weights
    logw = log(wN);

    % ----------------------------------------------------
    % precompute G
    
    logwopt = log(wopt);
    
    rmax = max(rT);
    G = -inf(rmax+1,mT,nT-1);
    G(1,:,:) = 0;
    G(2,:,nT-1) = logwopt(:,nT);
    
    for i = 1:mT
        ri = rT(i);
        for j = nT-1:-1:2
            wij = logwopt(i,j);
            for k = 2:ri+1
                b = G(k-1,i,j)+wij;
                a = G(k,i,j);
                if a > -inf || b > -inf
                    if a > b
                        G(k,i,j-1) = a + log(1+exp(b-a));
                    else
                        G(k,i,j-1) = b + log(1+exp(a-b));
                    end
                end
            end
        end
        
        for j = 1:nT-1
            for k = 1:rmax
                Gknum = G(k,i,j);
                Gkden = G(k+1,i,j);
                if isinf(Gkden)
                    G(k,i,j) = -1;
                else
                    G(k,i,j) = wopt(i,j)*exp(Gknum-Gkden)*((nT-j-k+1)/k);
                end
            end
            if isinf(Gkden)
                G(rmax+1,i,j) = -1;
            end
        end
    end
    
    % ----------------------------------------------------

    
else
    switch lower(cflag)
        case 'none'
            cndx = 1:numel(cN);
        case 'descend'
            [csort,cndx] = sort(cN(:),'descend');
        otherwise
            error('unknown cflag')
    end
end



% generate the inverse index for the row orders to facilitate fast
% sorting during the updating
irndxT = (1:mT).'; irndxT(rndxT) = irndxT;

% basic input checking
if rsort(1) > nT || rsort(mT) < 0 || csort(1) > mT || csort(nT) < 0 || any(rsort ~= round(rsort)) || any(csort ~= round(csort)), error('marginal entries invalid'), end

% compute the conjugate of c
cconjT = conjugate_local(csort,mT);

% get the running total of number of ones to assign
countT = sum(rsort);

% get the running total of sum of c squared
ccount2T = sum(csort.^2);
% get the running total of (2 times the) column marginals choose 2
ccount2cT = sum(csort.*(csort-1));
% get the running total of (6 times the) column marginals choose 3
ccount3cT = sum(csort.*(csort-1).*(csort-2));

% get the running total of sum of r squared
rcount2T = sum(rsort.^2);
% get the running total of (2 times the) row marginals choose 2
rcount2cT = sum(rsort.*(rsort-1));
% get the running total of (6 times the) row marginals choose 3
rcount3cT = sum(rsort.*(rsort-1).*(rsort-2));

% check for compatible marginals
if countT ~= sum(csort) || any(cumsum(rsort) > cumsum(cconjT)), error('marginal sums invalid'), end

% generate the correction for non-uniform
%if doW && isempty(wopt)

%	[~,~,wopt] = canonical(wN,wflag);
%	G = permute(AllWeights(wopt,rT),[3 1 2]);

% 	switch lower(wflag)
%
% 		case 'sinkhorn'
% 			tmpt = zeros(nT,nT); for k = 1:nT-1, for j = k+1:nT, tmpt(j,k) = 1./(nT-k); end, end, tmpt(nT,nT)=1;
% 			wopt = max(wN,eps(1));  % hack for zeros
% 			woptt = wopt.';
% 			tmpfix = ones(mT,nT);
%
% 			tmpfix0 = tmpfix;
% 			tol = inf;
% 			tol0 = 1e-6;
% 			while tol > tol0
% 				tmpfix = (1./(wopt*((mT./(woptt*tmpfix)).*tmpt)));
% 				tol = mean(abs(tmpfix(:)-tmpfix0(:)));
% 				tmpfix0 = tmpfix;
% 			end
% 			wopt = wopt.*tmpfix;
%
%         case 'l2once'
%             [~,~,wopt] = canonical(wN,'l2');
%             tmpt = bsxfun(@minus,sum(wopt,2),cumsum(wopt,2));
%             wopt(:,1:end-1) = wopt(:,1:end-1)./max(tmpt(:,1:end-1),eps(1));
%
% 		case 'gonly'
% 			doG = true;
% 			wopt = wN;
% 			G = permute(AllWeights(wopt,rT),[3 1 2]);
%
% 		case 'gl2'
% 			doG = true;
% 			[~,~,wopt] = canonical(wN,'l2');
% 			G = permute(AllWeights(wopt,rT),[3 1 2]);
%
% 		case 'gsinkhorn'
% 			doG = true;
% 			[~,~,wopt] = canonical(wN,'sinkhorn');
% 			G = permute(AllWeights(wopt,rT),[3 1 2]);
%
% 		case 'gsinkhorn-col'
% 			doG = true;
% 			[~,~,wopt] = canonical(wN,'sinkhorn-col');
% 			G = permute(AllWeights(wopt,rT),[3 1 2]);
%
% 		otherwise
% 			error('unknown wflag')
% 	end
%end

% initialize the memory
logQ = zeros(SampN,1);
logP = zeros(SampN,1);
if doA, AN = SampN; else AN = 1; end
alist = zeros(2,countT,AN);
% initialize the memory
M = csort(1)+3; % index 1 corresponds to -1; index 2 corresponds to 0, index 3 corresponds to 1, ..., index M corresponds to c(1)+1
S = zeros(M,nT);
SS = zeros(M,1);

eps0 = eps(0); % used to prevent divide by zero

%------------------------------------------------------%
%--------------- END: PREPROCESSING -------------------%
%------------------------------------------------------%

% loop over the number of samples
for SampLoop = 1:SampN
    
    %--------------- INITIALIZATION -----------------------%
    if doA, ALoop = SampLoop; else ALoop = 1; end
    
    % copy in initialization
    r = rT;
    rndx = rndxT;
    irndx = irndxT;
    
    cconj = cconjT;
    count = countT;
    ccount2 = ccount2T;
    ccount2c = ccount2cT;
    ccount3c = ccount3cT;
    rcount2 = rcount2T;
    rcount2c = rcount2cT;
    rcount3c = rcount3cT;
    m = mT;
    n = nT;
    
    % initialize
    place = 0; % most recent assigned column in alist
    logq = 0; % running log probability
    logp = 0;
    
    %------------------------------------------------------%
    %--------------- START: COLUMN-WISE SAMPLING ----------%
    %------------------------------------------------------%
    
    %-------- loop over columns ------------%
    for c1 = 1:nT
        
        %while count > 0
        
        %-----------------------------------------------------------------%
        %------------- START: SAMPLE THE NEXT "COLUMN" -------------------%
        %-----------------------------------------------------------------%
        
        % remember the starting point for this columns
        placestart = place + 1;
        
        %--------------------------------
        % sample a col
        %--------------------------------
        
        label = cndx(c1); % current column label
        
        colval = csort(c1); % current column value
        
        if colval == 0 || count == 0, break, end
        
        % update the conjugate
        for i = 1:colval
            cconj(i) = cconj(i)-1;
        end
        % update the number of columns remaining
        n = n - 1;
        
        %------------ DP initialization -----------
        
        smin = colval;
        smax = colval;
        cumsums = count;
        % update the count
        count = count - colval;
        % update running total of sum of c squared
        ccount2 = ccount2 - colval^2;
        % update the remaining (two times the) sum of column sums choose 2
        ccount2c = ccount2c - colval*(colval-1);
        % update the remaining (six times the) sum of column sums choose 3
        ccount3c = ccount3c - colval*(colval-1)*(colval-2);
        
        cumconj = count;
        
        SS(colval+3) = 0;
        SS(colval+2) = 1;
        SS(colval+1) = 0;
        
        % get the constants for computing the probabilities
        % it is faster to compute them all, than to check pflag
        d = ccount2c/count^2;
        if (count == 0) || (m*n == count)
            weightA = 0;
        else
            weightA = m*n/(count*(m*n-count));
            weightA = weightA*(1-weightA*(ccount2-count^2/n))/2;
        end
        
        d2 = ccount2c/(2*count^2+eps0) + ccount2c/(2*count^3+eps0) + ccount2c^2/(4*count^4+eps0);
        d3 = -ccount3c/(3*count^3+eps0) + ccount2c^2/(2*count^4+eps0);
        d22 = ccount2c/(4*count^4+eps0) + ccount3c/(2*count^4+eps0) - ccount2c^2/(2*count^5+eps0);
        
        %----------- dynamic programming ----------
        SSS = 0;
        % loop over (remaining and sorted descending) rows in reverse
        for i = m:-1:1
            
            % get the value of this row and use it to compute the
            % probability of a 1 for this row/column pair
            rlabel = rndx(i);
            val = r(rlabel);
            if ptype == 1
                % canfield
                %weight = val*exp(weightA*((val-1-count/m)^2-(val-count/m)^2));
                %p = weight./(n+1-val+weight);
                p = val*exp(weightA*(1-2*(val-count/m)));
                p = p./(n+1-val+p);
                q = 1-p;
            elseif ptype == 2
                % greenhill
                % q = 1/(1+val*exp(-(val-1)d2*(-2*(val-1))+d3*(-3*(val-1)*(val-2))+d22*((rcount2c-2*(val-1))^2-rcount2c^2)));
                q = 1/(1+val*exp((2*d2+3*d3*(val-2)+4*d22*(rcount2c-val+1))*(val-1)));
                p = 1-q;
            else
                % never get here
                p = 0; q = 0; % helps compiler
            end
            
            % incorporate weights
            if doW && n > 0 && val > 0
                Gk = G(val,rlabel,c1);
                if Gk < 0
                    q = 0;
                else
                    p = p*Gk;  
                end
            end
            
            % update the feasibility constraints
            cumsums = cumsums - val;
            cumconj = cumconj - cconj(i);
            
            sminold = smin;
            smaxold = smax;
            
            % incorporate the feasibility constraints into bounds on the
            % running column sum
            smin = max(0,max(cumsums-cumconj,sminold-1));
            smax = min(smaxold,i-1);
            
            % DP iteration
            SSS = 0;
            
            SS(smin+1) = 0;  % no need to set S(1:smin) = 0, since it is not accessed
            for j = smin+2:smax+2
                a = SS(j)*q;
                b = SS(j+1)*p;
                apb = a + b;
                SSS = SSS + apb;
                SS(j) = apb;
                S(j,i) = b/(apb+eps0);
            end
            SS(smax+3) = 0;  % no need to set S(smax+4:end) = 0, since it is not accessed
            
            % check for impossible
            if SSS <= 0, break, end
            
            % normalize to prevent overflow/underflow
            for j = smin+2:smax+2
                SS(j) = SS(j) / SSS;
            end
            
        end
        
        % check for impossible
        if SSS <= 0, logp = -inf; break, end
        
        %----------- sampling ----------
        j = 2; % running total (offset to match indexing offset)
        jmax = colval + 2;
        if j < jmax % skip assigning anything when colval == 0
            if doIN
                for i = 1:m
                    % get the transition probability of generating a one
                    p = S(j,i);
                    % get the current row
                    rlabel = rndx(i);
                    if bIN(rlabel,label)
                        
                        % if we have a generated a one, then decrement the current
                        % row total
                        val = r(rlabel);
                        r(rlabel) = val-1;
                        
                        % test stuff
                        rcount2 = rcount2 - 2*val + 1;
                        rcount2c = rcount2c - 2*val + 2;
                        rcount3c = rcount3c - 3*(val-1)*(val-2);
                        %
                        
                        % record the entry and update the log probability
                        place = place + 1;
                        logq = logq + log(p);
                        if doW, logp = logp + logw(rlabel,label); end
                        alist(1,place,ALoop) = rlabel;
                        alist(2,place,ALoop) = label;
                        j = j + 1;
                        % the next test is not necessary, but seems more efficient
                        % since all the remaining p's must be 0
                        if j == jmax, break, end
                    else
                        logq = logq + log(1-p);
                    end
                end
            else
                for i = 1:m
                    % get the transition probability of generating a one
                    p = S(j,i);
                    if rand <= p
                        
                        % if we have a generated a one, then decrement the current
                        % row total
                        rlabel = rndx(i);
                        val = r(rlabel);
                        r(rlabel) = val-1;
                        
                        rcount2 = rcount2 - 2*val + 1;
                        rcount2c = rcount2c - 2*val + 2;
                        rcount3c = rcount3c - 3*(val-1)*(val-2);
                        
                        % record the entry and update the log probability
                        place = place + 1;
                        logq = logq + log(p);
                        if doW, logp = logp + logw(rlabel,label); end
                        alist(1,place,ALoop) = rlabel;
                        alist(2,place,ALoop) = label;
                        j = j + 1;
                        % the next test is not necessary, but seems more efficient
                        % since all the remaining p's must be 0
                        if j == jmax, break, end
                    else
                        logq = logq + log(1-p);
                    end
                end
            end
        end
        
        %-----------------------------------------------------------------%
        %------------- END: SAMPLE THE NEXT "COLUMN" ---------------------%
        %-----------------------------------------------------------------%
        
        if count == 0, break, end
        
        %-----------------------------------------------
        % everything is updated except the sorting
        %-----------------------------------------------
        
        %-----------------------------------------------------------------%
        %------------- START: RESORT THE NEW ROW SUMS --------------------%
        %-----------------------------------------------------------------%
        
        % re-sort the assigned rows
        
        % this code block takes each row that was assigned to the list
        % and either leaves it in place or swaps it with the last row
        % that matches its value; this leaves the rows sorted (descending)
        % since each row was decremented by only 1
        
        % looping in reverse ensures that least rows are swapped first
        for j = place:-1:placestart
            % get the row label and its new value (old value -1)
            k = alist(1,j,ALoop);
            val = r(k);
            % find its entry in the sorting index
            irndxk = irndx(k);
            % look to see if the list is still sorted
            irndxk1 = irndxk + 1;
            if irndxk1 > m || r(rndx(irndxk1)) <= val
                % no need to re-sort
                continue;
            end
            % find the first place where k can be inserted
            irndxk1 = irndxk1 + 1;
            while irndxk1 <= m && r(rndx(irndxk1)) > val
                irndxk1 = irndxk1 + 1;
            end
            irndxk1 = irndxk1 - 1;
            % now swap irndxk and irndxk1
            rndxk1 = rndx(irndxk1);
            rndx(irndxk) = rndxk1;
            rndx(irndxk1) = k;
            irndx(k) = irndxk1;
            irndx(rndxk1) = irndxk;
        end
        
        %-----------------------------------------------------------------%
        %------------- END: RESORT THE NEW ROW SUMS ----------------------%
        %-----------------------------------------------------------------%
        
        % r(rndx(rndx1:rndxm)) is sorted descending and has exactly those
        % unassigned rows
        % rndx(rndx1:rndxm) still gives the labels of those rows
        % rndx(irndx(k)) = k
        %
        % c(c1+1:cn) is sorted descending and has exactly those unassigned columns
        % cndx(c1+1:cn) still gives the labels of those columns
        %
        % m, n, count, ccount2, ccount2c are valid for the remaining rows, cols
        
    end
    
    logQ(SampLoop) = logq;
    logP(SampLoop) = logp;
    
end

%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%------------------ END OF MAIN FUNCTION ---------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%
%-------------------------------------------------------------------------%


% helper function (just to keep everything together... not for efficiency,
% since it is only called once)

function cc = conjugate_local(c,n)
% function cc = conjugate(c,n)
%
% let c(:) be nonnegative integers
% cc(k) = sum(c >== k)  for k = 1:n

cc = zeros(n,1);

%c = min(c,n);

for j = 1:numel(c)
    k = c(j);
    if k >= n
        cc(n) = cc(n) + 1;
    elseif k >= 1
        cc(k) = cc(k) + 1;
    end
end

s = cc(n);
for j = n-1:-1:1
    s = s + cc(j);
    cc(j) = s;
end

%-----------------------------------


