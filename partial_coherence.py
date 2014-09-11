import numpy as np
import bagOfns as bg
import functools


def make_inverse(x_array_mask, L_mapping, verbose = 'v'):
    if verbose == 'v':
        print 'making the b_mask...' 
    b_mask = L_mapping(x_array_mask)
    b_mask = np.abs(b_mask) > 0.0

    if verbose == 'v':
        print '\nMapping the problem to 2-D and 1-D real vectors...'
        progress = True
    else :
        progress = False
    A, x, b = bg.matrify(x_array_mask, L_mapping, bmask = b_mask, dt=np.float64, progress=progress)
    
    if verbose == 'v':
        print '\nPerforming singular value decomposition...' 
    U, s, Vt = np.linalg.svd(A)

    # construct the psuedo inverse of A
    S_inv = np.zeros((A.shape[0], A.shape[1]), dtype=A.dtype)
    S_inv[:s.shape[0], :s.shape[0]] = np.diag(1. / s)
    S_inv = S_inv.T
    #
    A_inv = np.dot(Vt.T, np.dot(S_inv, U.T)) 
    
    if verbose == 'v':
        error = bg.l2norm(np.identity(A.shape[1], dtype=A.dtype), np.dot(A_inv, A))
        print '\n|| I - A_inv . A || = ', error

    L_inv = functools.partial(A_inv_to_L_inv, A_inv, x.array, b.vect)

    return L_inv

def A_inv_to_L_inv(A_inv, fnc_array, fnc_vect, b):
    b_vect = fnc_vect(b)
    x = np.dot(A_inv, b_vect)
    x_array = fnc_array(x)
    return x_array
