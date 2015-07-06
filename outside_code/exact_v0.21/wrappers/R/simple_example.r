
source("exact.r")

# Darwin's finches dataset
a <- c(14,13,14,10,12,2,10,1,10,11,6,2,17)  # row sums
b <- c(4,4,11,10,10,8,9,10,8,9,3,10,4,7,9,3,3)  # column sums
matrix_type <- 0  # 0: binary, 1: nonnegative integer
        
# Count the number of matrices with row sums a and column sums b
number <- count(a,b,matrix_type)
# (or, to specify the filenames to be used:)
# number <- count(a,b,matrix_type,'input.txt','table.bin')
cat('Number of matrices =',number,'\n')

# Draw some random samples from the uniform distribution on this set of matrices
number_of_samples <- 3
x <- sample(a,b,number_of_samples)
# (or, to specify the filenames to be used:)
# x <- sample(a,b,number_of_samples,'input.txt','table.bin','output.txt')
    

# Display samples
for (i in 1:number_of_samples) {
    write.table(x[,,i], row.names = FALSE, col.names = FALSE)
    cat('\n')
}

