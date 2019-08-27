import numpy as np
from scipy import linalg
from scipy.sparse import linalg as slinalg
from sklearn import covariance

import covadapt.matrix


def eigh_lw_samples(samples, grad_samples, n_eigs):
    class LinOp(slinalg.LinearOperator):
        def __init__(self, cov):
            super().__init__(shape=(cov.n, cov.n), dtype=np.float64)
            self._cov = cov

        def _matvec(self, v):
            out = np.empty_like(v)
            self._cov.matmult(v, out)
            return out

    shrinkage = covariance.ledoit_wolf_shrinkage(samples)
    cov = covadapt.matrix.LedoitWolf(samples, shrinkage)
    vals, vecs = slinalg.eigsh(LinOp(cov), n_eigs, which='LM')

    return vals, vecs


def eigh_lw_samples_grads(samples, grad_samples, n_eigs, n_eigs_grad, n_final):
    k, n = samples.shape
    vals, vecs = eigh_lw_samples(samples, None, n_eigs)
    vals_grad, vecs_grad = eigh_lw_samples(grad_samples, None, n_eigs_grad)

    vals = np.concatenate([vals, 1 / vals_grad])
    vecs = np.concatenate([vecs, vecs_grad], axis=1)
    log_vals = np.log(vals)

    def matvec(v):
        return vecs @ (log_vals * (vecs.T @ v))

    linop = slinalg.LinearOperator(shape=(n, n), dtype=np.float64, matvec=matvec)

    vals, vecs = slinalg.eigsh(linop, n_final, which='LM')
    vals = np.exp(vals)
    return vals, vecs
