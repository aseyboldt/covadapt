import time

import pymc3 as pm
from pymc3.step_methods.hmc.quadpotential import QuadPotential, _WeightedVariance
import theano
from pymc3.math import floatX
import numpy as np

from covadapt.matrix import Eigvals, DiagScaled


class EigvalsAdapt(QuadPotential):
    def __init__(self, n, initial_mean, eigvalsfunc, eigvalsfunc_kwargs, initial_diag=None, initial_weight=0,
                 adaptation_window=303, dtype=None, display=False):
        """Set up a diagonal mass matrix."""
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError('Initial diagonal must be one-dimensional.')
        if initial_mean.ndim != 1:
            raise ValueError('Initial mean must be one-dimensional.')
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError('Wrong shape for initial_diag: expected %s got %s'
                             % (n, len(initial_diag)))
        if len(initial_mean) != n:
            raise ValueError('Wrong shape for initial_mean: expected %s got %s'
                             % (n, len(initial_mean)))

        if dtype is None:
            dtype = theano.config.floatX

        if initial_diag is None:
            initial_diag = np.ones(n, dtype=dtype)
            initial_weight = 1

        self._eigvalsfunc = eigvalsfunc
        self._eigvalsfunc_kwargs = eigvalsfunc_kwargs

        self.dtype = dtype
        self._n = n
        self._var = np.array(initial_diag, dtype=self.dtype, copy=True)
        self._stds = np.sqrt(initial_diag)
        self._inv_stds = floatX(1.) / self._stds
        self._foreground_var = _WeightedVariance(
            self._n, initial_mean, initial_diag, initial_weight, self.dtype)
        self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
        self._n_samples = 0
        self.adaptation_window = adaptation_window
        self._samples = []
        self._grads = []
        
        # Work around numba bug #3569
        vecs = np.ones((n, 2)).copy('F')
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

    def _update_from_weightvar(self, weightvar):
        diag = np.empty(self._n)
        weightvar.current_variance(out=diag)
        self._cov.update_diag(diag)
        self._cov_inv = self._cov.inv()
    
    def _update_from_samples_grads(self, samples, grads):
        if self._display:
            print("n_samples", len(samples))
        
        samples = np.array(samples)
        grads = np.array(grads)
        
        samples[...] -= samples.mean(0)[None, :]
        grads[...] -= grads.mean(0)[None, :]
        samples[...] /= self._cov._diag_sqrt[None, :]
        grads[...] *= self._cov._diag_sqrt[None, :]
        
        #print(samples)
        #print(grads)
        #print("svdvals samples", linalg.svdvals(samples.T))
        #print("svdvals grads", linalg.svdvals(grads.T))
        start = time.time()
        vals, vecs = self._eigvalsfunc(samples, grads, **self._eigvalsfunc_kwargs)
        end = time.time()
        if self._display:
            print("Finding eigenvalues took %ss" % (end - start))
            print("eigvals", vals)
        #others = vals[n_eigs - 2 : n_eigs + 2].mean()
        #print("others", others)
        others = 1.
        #print("others", others)
        #print()
        inner = Eigvals(np.array(vals), np.array(vecs, order='F'), others)
        self._cov.update_inner(inner)
        self._cov_inv = self._cov.inv()

    def update(self, sample, grad, tune):
        """Inform the potential about a new sample during tuning."""
        if not tune:
            return

        window = self.adaptation_window

        if self._n_samples > window:
            self._samples.append(sample)
            self._grads.append(grad)


        self._foreground_var.add_sample(sample, weight=1)
        self._background_var.add_sample(sample, weight=1)
        

        if self._n_samples > 0 and self._n_samples % window == 0:
            self._update_from_weightvar(self._foreground_var)

            self._foreground_var = self._background_var
            self._background_var = _WeightedVariance(self._n, dtype=self.dtype)
            
            if self._n_samples >= 400:
                self._update_from_samples_grads(self._samples, self._grads)
                self._samples.clear()
                self._grads.clear()

        self._n_samples += 1
