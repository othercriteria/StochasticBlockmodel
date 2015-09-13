function samples = sample(p,q,k,input_filename,table_filename,output_filename)
% Draw samples from the uniform distribution on matrices with specified margins.
%
% INPUTS:
% p = (m x 1) vector of nonnegative numbers
% q = (n x 1) vector of nonnegative numbers, such that sum(p)==sum(q)
% k = number of samples to draw
% input_filename = filename of input data (this must be the same one used in count.m).
% table_filename = filename of saved binary data (same one used in count.m).
% output_filename = string that will be used as a filename for output data.
%
% OUTPUT:
% samples = (m x n x k) array of sampled (m x n) matrices
%
% (Note: This is a wrapper for the executable sample.exe.)


if nargin<6; output_filename = '._____output_____.dat'; end
if nargin<5; table_filename = '._____table_____.bin'; end
if nargin<4; input_filename = '._____input_____.dat'; end

if ~exist(input_filename)
    error('Input file %s does not exist. You must call count.m first.',input_filename);
end
if ~exist(table_filename)
    error('Table file %s does not exist. You must call count.m first.',table_filename);
end

m = length(p);
n = length(q);

% execute
command = sprintf('sample.exe %s %s %s %d 1',input_filename,table_filename,output_filename,k);
[status,output] = system(command);
if (status~=0), error(output); end

% read samples from file
data = load(output_filename);

% rearrange samples into 3-D array
samples = permute(reshape(data',[n,m,k]),[2,1,3]);











