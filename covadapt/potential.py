import time
import collections

import pymc3 as pm
from pymc3.step_methods.hmc.quadpotential import QuadPotential, _WeightedVariance
import theano
from pymc3.math import floatX
import numpy as np
from scipy import linalg

from covadapt.matrix import Eigvals, DiagScaled, Diag


class EigvalsAdapt(QuadPotential):
    def __init__(
        self,
        n,
        initial_mean,
        estimators,
        initial_diag=None,
        initial_weight=0,
        adaptation_window=300,
        dtype=None,
        display=False,
        cutoff=0.7,
    ):
        """Set up a diagonal mass matrix."""
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError("Initial diagonal must be one-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError(
                "Wrong shape for initial_diag: expected %s got %s"
                % (n, len(initial_diag))
            )
        if len(initial_mean) != n:
            raise ValueError(
                "Wrong shape for initial_mean: expected %s got %s"
                % (n, len(initial_mean))
            )

        if dtype is None:
            dtype = theano.config.floatX

        if initial_diag is None:
            initial_diag = np.ones(n, dtype=dtype)
            initial_weight = 1

        self._estimators = estimators
        self._cutoff = cutoff

        self._skip_first = 70
        self._n_diag_only = 300
        self._n_test = 100

        self.dtype = dtype
        self._n = n
        self._var = np.array(initial_diag, dtype=self.dtype, copy=True)
        self._stds = np.sqrt(initial_diag)
        self._inv_stds = floatX(1.0) / self._stds
        self._foreground_var = _WeightedVariance(
            self._n, initial_mean, initial_diag, initial_weight, self.dtype
        )
        self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
        self._n_samples = 0
        self.adaptation_window = adaptation_window
        self._samples = collections.deque([], adaptation_window)
        self._grads = collections.deque([], adaptation_window)

        # Work around numba bug #3569
        vecs = np.ones((n, 2))
        inner = Eigvals(np.ones((2,)), vecs, 1)
        self._cov = DiagScaled(initial_diag, inner)
        self._cov_inv = self._cov.inv()
        self._display = display

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        if out is None:
            out = np.empty_like(x)
        self._cov.matmult(x, out)
        return out

    def energy(self, x, velocity=None):
        """Compute kinetic energy at a position in parameter space."""
        if velocity is not None:
            return 0.5 * x.dot(velocity)
        return 0.5 * self._cov.quadform(x)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at a position in parameter space."""
        self.velocity(x, out=v_out)
        return self.energy(x, v_out)

    def random(self):
        """Draw random value from QuadPotential."""
        out = np.empty(self._n)
        self._cov_inv.draw(out)
        return out

    def _update_diag(self, weightvar):
        diag = np.empty(self._n)
        weightvar.current_variance(out=diag)
        self._cov.update_diag(diag)
        self._cov_inv.update_diag(1 / diag)

    def _update_offdiag(self, samples, grads):
        if self._display:
            print("n_samples", len(samples))

        start = time.time()
        stds, vals, vecs = eigvals_from_window(
            samples, grads, self._n_test, self._estimators, self._cutoff
        )
        end = time.time()
        if self._display:
            print("Finding eigenvalues took %ss" % (end - start))
            print("eigvals", vals)

        if len(vals) > 0:
            inner = Eigvals(np.array(vals), np.array(vecs, order="C"), 1.0)
            self._cov = DiagScaled(stds ** 2, inner)
            self._cov_inv = self._cov.inv()
        else:
            self._cov = Diag(stds ** 2)
            self._cov_inv = self._cov.inv()

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if not tune:
            return

        window = self.adaptation_window

        if self._n_samples > self._skip_first:
            self._samples.append(sample)
            self._grads.append(grad)

        if self._n_samples < self._n_diag_only:
            self._foreground_var.add_sample(sample, weight=1)
            self._background_var.add_sample(sample, weight=1)
            self._update_diag(self._foreground_var)
            if self._n_samples > 0 and self._n_samples % (window // 2) == 0:
                self._foreground_var = self._background_var
                self._background_var = _WeightedVariance(self._n, dtype=self.dtype)

        if (
            self._n_samples >= self._n_diag_only
            and (self._n_samples - self._n_diag_only) % (window // 3) == 0
        ):
            self._update_offdiag(list(self._samples), list(self._grads))

        self._n_samples += 1


def eigvals_from_window(samples, grads, n_test, estimators, cutoff):
    assert len(grads) == len(samples)
    assert n_test < len(samples)

    samples = np.array(samples)
    grads = np.array(grads)

    n_samples, n_dim = samples.shape

    stds = samples.std(0)
    mean = samples.mean(0)

    samples[...] -= mean[None, :]
    samples[...] /= stds[None, :]
    grads[...] *= stds[None, :]

    train_samples = samples[:-n_test]
    test_samples = samples[-n_test:]

    train_grads = grads[:-n_test]
    test_grads = grads[-n_test:]

    eigvars = []
    eigvecs = []
    for func in estimators:
        vars, vecs = func(train_samples, train_grads)
        vecs = vecs.T
        for var, vec in zip(vars, vecs):
            if np.abs(np.log(var)) > cutoff:
                eigvars.append(var)
                eigvecs.append(vec)

    if len(eigvars) == 0:
        return stds, [], []

    eigvecs = np.array(eigvecs)
    _, S, V = linalg.svd(eigvecs, full_matrices=False)
    vecs = V[S > 0.95]

    projected_test_samples = test_samples @ vecs.T

    _, svd, inner_eigvals = linalg.svd(projected_test_samples, full_matrices=False)
    test_eigvals = svd ** 2 / n_test

    test_eigvecs = inner_eigvals @ vecs

    final_vars = []
    final_vecs = []
    for val, vec in zip(test_eigvals, test_eigvecs):
        if np.abs(np.log(val)) > cutoff:
            final_vars.append(val)
            final_vecs.append(vec)

    final_vecs = np.array(final_vecs).T
    final_vars = np.array(final_vars)

    return stds, final_vars, final_vecs
