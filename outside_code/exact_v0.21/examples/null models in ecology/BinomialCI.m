function [lower,upper] = BinomialCI(x,n,alpha,tol)
% Copyright (c) 2010, Matthew T. Harrison
% All rights reserved.

if nargin < 4, tol = 1e-8; end

p = x/n;
a = (1-alpha)/2;

if x == 0
    c1 = 0;
else
    low = 0;
    high = 1;
    c1 = p;
    while high-low > tol
        if 1-binocdf(x-1,n,c1) > a
            high = c1;
        else
            low = c1;
        end
        c1 = (low+high)/2;
    end
end

if x == n
    c2 = 1;
else
    low = 0;
    high = 1;
    c2 = p;
    while high-low > tol
        if binocdf(x,n,c2) > a
            low = c2;
        else
            high = c2;
        end
        c2 = (low+high)/2;
    end
end

lower = c1;
upper = c2;