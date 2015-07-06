


count <- function(a,b,matrix_type=0,input_filename='._____input_____.dat',
                                    table_filename='._____table_____.bin') {
# Count the number of matrices with row sums a and column sums b.
#
# INPUTS:
# a = vector of m nonnegative numbers
# b = vector of n nonnegative numbers, such that sum(a)==sum(b)
# matrix_type = 0: binary matrices, 1: nonnegative integer matrices
# input_filename = string that will be used as a filename for input data.
# table_filename = string that will be used as a filename for saving binary data.
#                  (This data will be used if you want sample, otherwise you can delete it.)
#
# OUTPUT:
# number = the number of matrices with row sums a and column sums b.
#          (This is a string, since it may be too large for normal R types.)
#
# (Note: This is a wrapper for the executable count.exe.)

m <- length(a)
n <- length(b)

# generate input file
sink(input_filename)
cat(m,n,matrix_type,'\n')
cat(as.character(a),'\n')
cat(as.character(b),'\n')
sink()

# run count.exe
arguments <- paste(input_filename, table_filename, '1')
output <- system2('count.exe', arguments, stdout=TRUE, stderr=TRUE)
status <- attr(output,"status")
if (!is.null(status)) stop(output)

return(output)
}



sample <- function(a,b,k,input_filename='._____input_____.dat',
                         table_filename='._____table_____.bin',
                         output_filename='._____output_____.dat') {
# Draw samples from the uniform distribution on matrices with specified margins.
#
# INPUTS:
# a = vector of m nonnegative numbers
# b = vector of n nonnegative numbers, such that sum(a)==sum(b)
# k = number of samples to draw
# input_filename = filename of input data (this must be the same one used in count()).
# table_filename = filename of saved binary data (same one used in count()).
# output_filename = string that will be used as a filename for output data.
#
# OUTPUT:
# samples = (m x n x k) array of sampled (m x n) matrices
#
# (Note: This is a wrapper for the executable sample.exe.)


if (!file.exists(input_filename))
    stop(sprintf('Input file %s does not exist. You must call count() first.',input_filename))
end
if (!file.exists(table_filename))
    stop(sprintf('Table file %s does not exist. You must call count() first.',table_filename))
end

m <- length(a)
n <- length(b)

# run sample.exe
arguments <- paste(input_filename, table_filename, output_filename, k, '1')
output <- system2('sample.exe', arguments, stdout=TRUE, stderr=TRUE)
status <- attr(output,"status")
if (!is.null(status)) stop(output)

# read samples from file
values = scan(output_filename, n = m*n*k, quiet = TRUE)

# rearrange samples into 3-D array
samples <- aperm(array(values, c(n,m,k)), c(2,1,3))

return(samples)
}













