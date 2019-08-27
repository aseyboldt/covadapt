# Better mass matrices for NUTS

This is an experimental implementation of a low rank approximation of
mass matrices for hamiltonian MCMC samplers, specifically for pymc3.

*This is for experimentation only! Do not use!*

# Example

```python
import covadapt.potential
import pymc3 as pm

with pm.Model() as model:
    pm.Normal('y', shape=100)

    pot = covadapt.potential.EigvalsL1Adapt(
        model.ndim, np.zeros(model.ndim), gamma=50,
        adaptation_window=200,
        n_eigs=3, n_grads=3,
    )
    step = pm.NUTS(potential=pot)
    trace = pm.sample(step=step, draws=1000, tune=2000, chains=4)
```

And an example that fails with the pymc3 standard sampler:
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

    pot = covadapt.potential.EigvalsL1Adapt(
        model.ndim, np.zeros(model.ndim), gamma=50,
        adaptation_window=200,
        n_eigs=3, n_grads=3,
    )
    step = pm.NUTS(potential=pot)
    trace = pm.sample(step=step, draws=1000, tune=2000, chains=4)
```
