import numba
import numpy as np
from scipy import linalg


class SPDMatrix:
    def matmult(self, v, out):
        ...

    def inv(self):
        ...
    
    def draw(self):
        ...

    def quadform(self, v):
        ...


spec = [
    ('n', numba.uint64),
    ('k', numba.uint64),
    ('_eigvecs', numba.float64[::, ::1]),
    ('_eigvals', numba.float64[::1]),
    ('_scale', numba.float64),
    ('_long_tmp', numba.float64[::1]),
    ('_short_tmp', numba.float64[::1]),
]

#@jitclass(spec)
class Eigvals(SPDMatrix):
    """"""
    def __init__(self, eigvals, eigvecs, others):
        self.n, self.k = eigvecs.shape
        self._eigvals = eigvals / others
        self._eigvecs = eigvecs
        self._scale = others
        self._short_tmp = np.zeros(self.k)
        self._long_tmp = np.zeros(self.n)

    def matmult(self, v, out):
        # (vecs @ np.diag(vals) @ vecs.T + (np.eye(n) - vecs @ vecs.T)) @ x

        # work around numba bug with size 0 arrays
        if self.k == 0:
            out[:] = v
            return

        np.dot(self._eigvecs.T, v, out=self._short_tmp)
        np.dot(self._eigvecs, self._short_tmp, out=self._long_tmp)
        self._long_tmp[:] -= v
        
        np.dot(self._eigvecs.T, v, out=self._short_tmp)
        self._short_tmp[:] *= self._eigvals
        np.dot(self._eigvecs, self._short_tmp, out=out)
        
        out[:] -= self._long_tmp
        out[:] *= self._scale
    
    def inv(self):
        eigs = 1 / (self._eigvals * self._scale)
        return Eigvals(eigs, self._eigvecs, 1 / self._scale)
    
    def quadform(self, v):
        self.matmult(v, self._long_tmp)
        return np.inner(v, self._long_tmp)

    def draw(self, out):
        v = np.random.randn(self.n)
        self.sqrt_matmult(v, out)
    
    def sqrt_matmult(self, v, out):
        if self.k == 0:
            out[:] = v
            return

        np.dot(self._eigvecs.T, v, out=self._short_tmp)
        np.dot(self._eigvecs, self._short_tmp, out=self._long_tmp)
        self._long_tmp[:] -= v
        
        np.dot(self._eigvecs.T, v, out=self._short_tmp)
        self._short_tmp[:] *= np.sqrt(self._eigvals)
        np.dot(self._eigvecs, self._short_tmp, out=out)

        out[:] -= self._long_tmp
        out[:] *= np.sqrt(self._scale)


spd_type_diag = numba.deferred_type()
        
spec = [
    ('n', numba.uint64),
    ('_diag', numba.float64[::1]),
    ('_diag_sqrt', numba.float64[::1]),
    ('_inner_spd', spd_type_diag),
    ('_tmp', numba.float64[::1]),
]

#@numba.jitclass(spec)
class DiagScaled(SPDMatrix):
    def __init__(self, diag, inner_spd):
        self.n = len(diag)
        self._diag = diag
        self._diag_sqrt = np.sqrt(diag)
        self._inner_spd = inner_spd
        self._tmp = np.empty_like(diag)

    def matmult(self, v, out):
        self._tmp[:] = v * self._diag_sqrt
        self._inner_spd.matmult(self._tmp, out=out)
        out[:] *= self._diag_sqrt
    
    def draw(self, out):
        self._inner_spd.draw(out)
        out[:] *= self._diag_sqrt
    
    def quadform(self, v):
        self.matmult(v, self._tmp)
        return np.inner(v, self._tmp)

    def inv(self):
        return DiagScaled(1 / self._diag, self._inner_spd.inv())
    
    def update_inner(self, inner):
        self._inner_spd = inner
    
    def update_diag(self, diag):
        self._diag = diag
        self._diag_sqrt = np.sqrt(diag)


def kondition_number(A, B):
    eigvals = linalg.eigvalsh(A, B)
    return np.sqrt(eigvals[-1] / eigvals[0])

