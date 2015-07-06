# Example usage of the Python implementation

from exact_pure import *

examples_to_run = [1.1,2.1]
number_of_samples = 10

for example in examples_to_run:
    # ==================================================================
    # Ecology co-occurrence matrices
    if example==1.1:
        title = "Darwin's finches"
        p = [14,13,14,10,12,2,10,1,10,11,6,2,17]
        q = [4,4,11,10,10,8,9,10,8,9,3,10,4,7,9,3,3]
        true_number = 67149106137567626

    if example==1.2:
        title = "Manly's lizards"
        p = [7,23,6,6,2,18,8,8,1,22,2,2,9,2,1,18,9,3,3,1]
        q = [13,4,9,4,3,2,3,4,4,5,2,5,2,10,10,10,7,6,6,3,3,11,8,11,6]
        true_number = 55838420515731001979319625577023858901579264
        
    # ==================================================================
    # Social networks
    if example==2.1:
        title = "Sampson's monks"
        p = [9,7,2,3,5,1,4,4,2,1,2,1,2,6,1,2,1,2] 
        q = [3,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3,3,3]
        true_number = 10328878906560943043606457551481096000
        
        
    
    # Summary
    print '============== Example %.1f =============='%example
    print title
    print 'p =',p
    print 'q =',q
    
    # Count the number of binary matrices with row sums p and column sums q
    print 'Counting...'
    number,table = count(p,q)

    # Verify that we got the correct answer
    print 'True =   ',true_number
    print 'Counted =',number
    assert(number==true_number)

    # Draw some random samples from the uniform distribution on this set of matrices
    print 'Sampling',number_of_samples,'matrices...'
    X = sample(p,q,number_of_samples,table)
    
    # Display the 1st sample
    print '1st sample ='
    for line in X[0]: print line
    print

    
    
# A simple check of the uniformity of the sampler
print '====== Uniformity of the samples ======'
p = [3,1,2]
q = [1,2,1,2]
k = 10000
number,table = count(p,q)
print 'Number of matrices =',number
print 'Sampling',k,'matrices...'
X = sample(p,q,k,table)
histogram = {}
for x in X:
    key = repr(x)
    if key in histogram: histogram[key] += 1
    else: histogram[key] = 1
print 'Histogram of sampled matrices:'
print '(index)  (count)  (P_empirical/P_uniform)'
for i,v in enumerate(histogram.values()):
    print i, v, number*v/float(k)


print 'Hit enter to finish...'
raw_input()










