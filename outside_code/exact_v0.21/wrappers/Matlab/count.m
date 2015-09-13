function number = count(p,q,matrix_type,input_filename,table_filename)
% Count the number of matrices with row sums p and column sums q.
%
% INPUTS:
% p = (m x 1) vector of nonnegative numbers
% q = (n x 1) vector of nonnegative numbers, such that sum(p)==sum(q)
% matrix_type = 0: binary matrices, 1: nonnegative integer matrices
% input_filename = string that will be used as a filename for input data.
% table_filename = string that will be used as a filename for saving binary data.
%                  (This data will be used if you want to sample, otherwise you can delete it.)
%
% OUTPUT:
% number = the number of matrices x such that all(sum(x,2)==p) and all(sum(x,1)==q').
%          (This is a string, since it may be too large for normal Matlab types.)
%
% (Note: This is a wrapper for the executable count.exe.)


if nargin<5; table_filename = '._____table_____.bin'; end
if nargin<4; input_filename = '._____input_____.dat'; end
if nargin<3; matrix_type = 0; end

m = length(p);
n = length(q);

% generate input file
file = fopen(input_filename, 'w');
fprintf(file,'%d %d %d\r\n',m,n,matrix_type);
fprintf(file,'%d ',p); fprintf(file,'\r\n');
fprintf(file,'%d ',q); fprintf(file,'\r\n');
fclose(file);

% execute
command = sprintf('count.exe %s %s 1',input_filename, table_filename);
[status,output] = system(command);
if (status~=0), error(output); end

number = output;

