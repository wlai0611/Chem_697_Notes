import numpy as np

def covariance(A):
    '''
    Calculate a matrix where each cell represents a pairwise covariance between two columns of A
    
    Parameters
    ----------
    A: numpy.array
      Each column represents a variable
    Returns
    -------
    numpy.array
    '''
    centered_columns = A - np.mean(A,axis=0)
    return np.dot(centered_columns.T, centered_columns)/(A.shape[0]-1)