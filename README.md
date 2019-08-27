# Better mass matrices for NUTS

This is an experimental implementation of a low rank approximation of
mass matrices for hamiltonian MCMC samplers, specifically for pymc3.

*This is for experimentation only! Do not use!*

## Install

Make sure you have pymc3 and numba installed. Then do

```
pip install git+https://github.com/aseyboldt/covadapt.git
```

## Usage

There are three different methods implemented for finding eigenvalues
of the posterior covariance:

- `covadapt.eigvals_lw.eigh_lw_samples` This uses only the samples
  in the last adaptation window, and finds the largest eigenvalues
  of the ledoit-wolf estimate of the covariance matrix.
- `covadapt.eigvals_reg.eigh_lw_samples_grad`. This uses samples
  and their gradients to get ledoit-wolf estimates of the covariance
  and the inverse covariance. The two estimates are then combined
  on the log scale, and the largest and smallest eigenvalues of the
  combined matrix are computed.
- `covadapt.eigvals_reg.eigh_regularized_grad`. This uses a version
  of regularized pca with l1 loss to find eigenvalues of the covariance
  based on samples and gradients in the adaptation window. Of the three
  methods this is by far the most experimental one.

You can use one of the three methods like this:

```python
pot = covadapt.potential.EigvalsAdapt(
    model.ndim,
    np.zeros(model.ndim),
    covadapt.eigvals_reg.eigh_regularized_grad,
    eigvalsfunc_kwargs=dict(
        n_eigs=3, 
        n_eigs_grad=3,
        gamma=50,
        gamma_grad=50,
    ),
    adaptation_window=200,
)

pot2 = covadapt.potential.EigvalsAdapt(
    model.ndim,
    np.zeros(model.ndim),
    covadapt.eigvals_lw.eigh_lw_samples,
    eigvalsfunc_kwargs=dict(
        n_eigs=6,
    ),
    adaptation_window=200,
)

pot3 = covadapt.potential.EigvalsAdapt(
    model.ndim,
    np.zeros(model.ndim),
    covadapt.eigvals_lw.eigh_lw_samples_grads,
    eigvalsfunc_kwargs=dict(
        n_eigs=6,
        n_eigs_grad=6,
        n_final=15,
    ),
    adaptation_window=200,
)
```


```python
import covadapt.potential
import pymc3 as pm

with pm.Model() as model:
    pm.Normal('y', shape=100)

    pot3 = covadapt.potential.EigvalsAdapt(
        model.ndim,
        np.zeros(model.ndim),
        covadapt.eigvals_lw.eigh_lw_samples_grads,
        eigvalsfunc_kwargs=dict(
            n_eigs=6,
            n_eigs_grad=6,
            n_final=15,
        ),
        adaptation_window=200,
    )
    step = pm.NUTS(potential=pot)
    trace = pm.sample(step=step, draws=1000, tune=2000, chains=4)
```

And a complete example that fails with the pymc3 standard sampler:
```python
n = 500

U = np.array([[1, 0, -3, 0, 0, 6] + [0] * (n - 6),
              [0, 5, 0, 3, -2, 0] + [0] * (n - 6)]).T

U = U / np.sqrt((U ** 2).sum(0))[None, :]
true_eigvals = U
Σ = np.diag([2000000, 0.00001])
cov = U @ Σ @ U.T + (np.eye(n) - U @ U.T)


with pm.Model() as model:
    pm.MvNormal('a', shape=n, mu=0, cov=cov)

    pot = covadapt.potential.EigvalsAdapt(
        model.ndim,
        np.zeros(model.ndim),
        covadapt.eigvals_lw.eigh_lw_samples_grads,
        eigvalsfunc_kwargs=dict(
            n_eigs=6,
            n_eigs_grad=6,
            n_final=15,
        ),
        adaptation_window=200,
    )
    step = pm.NUTS(potential=pot)
    trace = pm.sample(step=step, draws=1000, tune=2000, chains=4)
```
