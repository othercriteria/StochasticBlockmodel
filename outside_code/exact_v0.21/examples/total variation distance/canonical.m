function [a,b,abw,k] = canonical(w,flag,tol,maxiter,r,c)
% Copyright (c) 2012, Matthew T. Harrison
% All rights reserved.

[m,n] = size(w);

if nargin <6 || isempty(c)
	c = ones(1,n);
elseif size(c,1) ~= 1
	c = c(:).';
end
if nargin <5 || isempty(r)
	r = ones(m,1);
elseif size(r,2) ~= 1
	r = r(:);
end
if nargin <4 || isempty(maxiter)
    maxiter = inf;
end
if nargin <3 || isempty(tol)
    tol = 1e-8;
end
if nargin <2 || isempty(flag)
    flag = 'sinkhorn';
end

switch lower(flag)
    
    case 'sinkhorn'
              
		M = sum(w>0,1); N = sum(w>0,2);
		a = N./sum(w,2); a = a/mean(a);		
		b = M./sum(bsxfun(@times,a,w),1);
		
		if tol >= 0, a0 = a; b0 = b; end

        k = 0;
        tolcheck = inf;
        while k < maxiter && tolcheck > tol
            k = k + 1;
		
			a = N./sum(bsxfun(@times,b,w),2); a = a/mean(a);
			b = M./sum(bsxfun(@times,a,w),1);
			
			if tol >= 0
                tolcheck = sum(abs(a-a0))+sum(abs(b-b0));
				a0 = a; b0 = b;
			end
        end 
             
    case 'sinkhorn-col'
              
		w = fliplr(w);
		
		M = sum(w>0,1); N = cumsum(w>0,2);
		aa = N./cumsum(w,2);
		%aa = bsxfun(@rdivide,aa,mean(aa,1));
		b = M./sum(w.*aa,1); b = b / mean(b);
		a = aa(:,n);
				
		if tol >= 0, a0 = a; b0 = b; end

        k = 0;
        tolcheck = inf;
        while k < maxiter && tolcheck > tol
            k = k + 1;
		
			aa = N./cumsum(bsxfun(@times,b,w),2);
			%aa = bsxfun(@rdivide,aa,mean(aa,1));
			b = M./sum(w.*aa,1); b / mean(b);
			a = aa(:,n);
			
			if tol >= 0
                tolcheck = sum(abs(a-a0))+sum(abs(b-b0));
				a0 = a; b0 = b;
			end
		end
		%a = N(:,n)./sum(bsxfun(@times,b,w),2);
		
		w = fliplr(w);
		b = fliplr(b);
		
	case 'log'
		
		w0 = w > 0;
		M = sum(w0,1); N = sum(w0,2);
		logw = log(w+~w0);
		a = exp(-sum(logw,2)./N);
		b = exp(-sum(logw,1)./M);
		
	case 'entropy'
		
		w1 = (w > 0)./max(w,eps(0));
		a = sqrt(sum(w1,2)./sum(w,2)); a = a/mean(a);
		b = sqrt(sum(bsxfun(@rdivide,w1,a),1)./sum(bsxfun(@times,a,w),1));
		
		if tol >= 0, a0 = a; b0 = b; end

        k = 0;
        tolcheck = inf;
        while k < maxiter && tolcheck > tol
            k = k + 1;
		
			a = sqrt(sum(bsxfun(@rdivide,w1,b),2)./sum(bsxfun(@times,b,w),2)); a = a/mean(a);
			b = sqrt(sum(bsxfun(@rdivide,w1,a),1)./sum(bsxfun(@times,a,w),1));
			
			if tol >= 0
                tolcheck = sum(abs(a-a0))+sum(abs(b-b0));
				a0 = a; b0 = b;
			end
        end 
		
	case 'l2'
              
		w2 = w.^2;
		
		a = sum(w,2)./sum(w2,2); a = a/mean(a);		
		b = sum(bsxfun(@times,a,w),1)./sum(bsxfun(@times,a.^2,w2),1);
		
		if tol >= 0, a0 = a; b0 = b; end

        k = 0;
        tolcheck = inf;
        while k < maxiter && tolcheck > tol
            k = k + 1;
		
			a = sum(bsxfun(@times,b,w),2)./sum(bsxfun(@times,b.^2,w2),2); a = a/mean(a);
			b = sum(bsxfun(@times,a,w),1)./sum(bsxfun(@times,a.^2,w2),1);
			
			if tol >= 0
                tolcheck = sum(abs(a-a0))+sum(abs(b-b0));
				a0 = a; b0 = b;
			end
        end 
             	
    case 'l2p'
        
        w2 = w.^2;
        
        c = (1+w).^3;
		a = sum(w./c,2)./sum(w2./c,2); a = a/mean(a);		
        c = (1+bsxfun(@times,a,w)).^3;
		b = sum(bsxfun(@times,a,w)./c,1)./sum(bsxfun(@times,a.^2,w2)./c,1);
		
		if tol >= 0, a0 = a; b0 = b; end

        k = 0;
        tolcheck = inf;
        while k < maxiter && tolcheck > tol
            k = k + 1;
		
            c = (1+a*b.*w).^3;
			a = sum(bsxfun(@times,b,w)./c,2)./sum(bsxfun(@times,b.^2,w2)./c,2); a = a/mean(a);
            c = (1+a*b.*w).^3;
			b = sum(bsxfun(@times,a,w)./c,1)./sum(bsxfun(@times,a.^2,w2)./c,1);
			
			if tol >= 0
                tolcheck = sum(abs(a-a0))+sum(abs(b-b0));
				a0 = a; b0 = b;
			end
        end 
        
    case 'ratio'
        
        wz = w > 0;
        w(~wz) = eps(0);
        
        a = sqrt(sum(wz./w,2)./sum(w,2)); a = a/mean(a);
		b = sqrt(sum(wz./(bsxfun(@times,a,w)),1)./sum(bsxfun(@times,a,w),1));
		
		if tol >= 0, a0 = a; b0 = b; end

        k = 0;
        tolcheck = inf;
        while k < maxiter && tolcheck > tol
            k = k + 1;
		
			a = sqrt(sum(wz./(bsxfun(@times,b,w)),2)./sum(bsxfun(@times,b,w),2)); a = a/mean(a);
            b = sqrt(sum(wz./(bsxfun(@times,a,w)),1)./sum(bsxfun(@times,a,w),1));
			
			if tol >= 0
                tolcheck = sum(abs(a-a0))+sum(abs(b-b0));
				a0 = a; b0 = b;
			end
        end 
        
        
	case 'barvinok'

		s = log(r/n);
		t = log(c/m);

		M = w.*(exp(s)*exp(t));
		M = M ./ (1+M);

		sMr = sum(M,2)-r;
		sMc = sum(M,1)-c;

		tolcheck = sum(abs(sMr))+sum(abs(sMc));

		alpha = .01;
		
		while tolcheck > tol
    
			s = s - alpha*sMr;
			t = t - alpha*sMc;
    
			M = w.*(exp(s)*exp(t));
			M = M ./ (1+M);
    
			sMr = sum(M,2)-r;
			sMc = sum(M,1)-c;
    
			tolcheck = sum(abs(sMr))+sum(abs(sMc));
		end
		
		a = exp(s);
		b = exp(t);
		
    otherwise
        
        error('unknown flag')
end

if nargout > 2, abw = a*b.*w; end
