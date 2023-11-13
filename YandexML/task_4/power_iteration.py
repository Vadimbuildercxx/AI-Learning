import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    
    vecs = []
    ls = []
    vecs.append(np.ones((data.shape[0], 1)))
    for i in range(1, num_steps):
        dot = np.dot(data, vecs[i-1])
        # xTx1 = vecs[i-1].T.dot(vecs[i])
        # xTx2 = vecs[i-1].T.dot(vecs[i-1])

        ls.append(np.divide(dot[0], vecs[i-1][0]))
        vecs.append(dot/ np.linalg.norm(dot))


    return float(ls[-1][0]), vecs[-1].squeeze()