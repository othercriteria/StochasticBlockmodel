
from exact import *

# Darwin's finches dataset
p = [14,13,14,10,12,2,10,1,10,11,6,2,17] # row sums
q = [4,4,11,10,10,8,9,10,8,9,3,10,4,7,9,3,3] # column sums
matrix_type = 0  # 0: binary, 1: nonnegative integer
        
# Count the number of matrices with row sums p and column sums q
number = count(p,q,matrix_type)
# (or, to specify the filenames to be used:)
# number = count(p,q,matrix_type,'input.txt','table.bin')
print number

# Draw some random samples from the uniform distribution on this set of matrices
number_of_samples = 3
X = sample(p,q,number_of_samples)
# (or, to specify the filenames to be used:)
# X = sample(p,q,number_of_samples,'input.txt','table.bin','output.txt')

# Display samples
for x in X:
    for line in x: print line
    print
    
