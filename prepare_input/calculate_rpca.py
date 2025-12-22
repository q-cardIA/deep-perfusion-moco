import numpy as np

def rpca(M,lam):
    """
    Cian Scannell - Oct-2017
    computes rpca separation of M into L and S using the parameter lam

    this uses the alternating directions augmented method of multipliers

    :param M: 3D matrix of shape (nr_x_coord, nr_y_coord, nr_image_frames)
    :param lam: trade-off parameter; default = 1/sqrt(nr_x_coord*nr_y_coord)
    :return: L & S matrix of shape (nr_x_coord, nr_y_coord, nr_image_frames)
    """

    # Get dimensions and reshape
    Nr = M.shape[0]
    Nc = M.shape[1]
    Nt = M.shape[2]
    M = M.reshape(Nr*Nc,Nt)

    Y = M / np.maximum(np.linalg.norm(M,2), np.linalg.norm(M,np.inf) / lam)         

    mu = 1/ (np.linalg.norm(M,2))       
    rho = 1.6                          

    S = np.zeros((Nr*Nc,Nt))

    error = 10
    count = 0
    while error > 1e-7:
        U,sig,V = np.linalg.svd(M-S+Y/mu, full_matrices=False)
        L = np.dot(U, np.dot(np.diag(soft_thres(sig, 1/mu)), V))
        S = soft_thres(M-L+Y/mu, lam/mu)
        Y = Y + mu*(M-L-S)
        mu = mu*rho
        error = np.linalg.norm(M-L-S,'fro') / np.linalg.norm(M,'fro')       
        count += 1

    L = L.reshape(Nr,Nc,Nt)
    S = S.reshape(Nr,Nc,Nt)

    return L,S

def soft_thres(x,eps):
    """
    # Cian Scannell - Oct-2017

    # soft thresholds a matrix x at the eps level
    # i.e ST(x,eps)_ij = sgn(x_ij) max(|x_ij| - eps, 0)

    :param x: matrix x
    :param eps: value
    :return:
    """
    a = np.sign(x)
    b = np.maximum((np.fabs(x) - eps), 0) 
    return np.multiply(a,b)