
import os

def count(p,q,matrix_type=0,input_filename='._____input_____.dat', 
                            table_filename='._____table_____.bin'):
    # Count the number of matrices with row sums p and column sums q.
    #
    # INPUTS:
    # p = list of m nonnegative numbers
    # q = list of n nonnegative numbers, such that sum(p)==sum(q)
    # matrix_type = 0: binary matrices, 1: nonnegative integer matrices
    # input_filename = string that will be used as a filename for input data.
    # table_filename = string that will be used as a filename for saving binary data.
    #                  (This data will be used if you want to sample, otherwise you can delete it.)
    #
    # OUTPUT:
    # number = the number of matrices with row sums p and column sums q.
    #
    # (Note: This is a wrapper for the executable count.exe.)

    m = len(p)
    n = len(q)

    # generate input file
    file = open(input_filename, 'w')
    file.write('%d %d %d\n'%(m,n,matrix_type))
    file.write(('%d '*m)%tuple(p)); file.write('\n')
    file.write(('%d '*n)%tuple(q)); file.write('\n')
    file.close()

    # execute
    command = 'count.exe %s %s 1'%(input_filename, table_filename)
    (out,inn,err) = os.popen3(command)
    error_text = err.read().strip()
    if error_text:
        raise Exception,'Error in system call: "%s"\n%s'%(command,error_text)
    output = inn.read().rstrip('\n')

    return int(output)

    
def sample(p,q,k,input_filename='._____input_____.dat',
                 table_filename='._____table_____.bin',
                 output_filename='._____output_____.dat'):
    # Draw samples from the uniform distribution on matrices with specified margins.
    #
    # INPUTS:
    # p = (m x 1) vector of nonnegative numbers
    # q = (n x 1) vector of nonnegative numbers, such that sum(p)==sum(q)
    # k = number of samples to draw
    # input_filename = filename of input data (this must be the same one used in count.m).
    # table_filename = filename of saved binary data (same one used in count.m).
    # output_filename = string that will be used as a filename for output data.
    #
    # OUTPUT:
    # samples = (m x n x k) array of sampled (m x n) matrices
    #
    # (Note: This is a wrapper for the executable sample.exe.)

    if not os.path.exists(input_filename):
        raise Exception,'Input file %s does not exist. You must call count.m first.'%input_filename
    if not os.path.exists(table_filename):
        raise Exception,'Table file %s does not exist. You must call count.m first.'%table_filename

    m = len(p)
    n = len(q)

    # execute
    command = 'sample.exe %s %s %s %d 1'%(input_filename,table_filename,output_filename,k)
    (out,inn,err) = os.popen3(command)
    error_text = err.read().strip()
    if error_text:
        raise Exception,'Error in system call: "%s"\n%s'%(command,error_text)
    
    # read samples from file
    file = open(output_filename, 'r')
    contents = file.read().strip()
    file.close()
    
    samples = []
    for chunk in contents.split('\n\n'):
        sample = []
        for line in chunk.split('\n'):
            sample.append([int(item) for item in line.split()])
        samples.append(sample)
        

    return samples









