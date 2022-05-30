# Better mass matrices for NUTS

This is an experimental implementation of a low rank approximation of
mass matrices for hamiltonian MCMC samplers, specifically for pymc3.

*This is for experimentation only! Do not use for actual work (yet)!*

But feel welcome to try it out, and tell me how it worked for your models!

## Install

```
pip install git+https://github.com/aseyboldt/covadapt.git
```

## Usage

See the `notebooks/covadapt_intro.ipynb` notebook.

## (Draft of an) Overview

When we use precondidioning with a mass matrix to improve performance of HMC
based on previous draws, we often ignore information that we already computed:
the gradients of the posterior density at those samples. But those gradients
contain a lot of information about the posterior geometry and as such also
about possible preconditioners. If for example we assume that the posterior is
an $n$-dimensional normal distribution, then knowing the gradient at $n + 1$
locations identifies the covariance matrix – and as such the optimal
preconditioner of the posterior – *exactly*.

We can evaluate a precondition matrix $\hat{\Sigma}$ by thinking of it and a
mean $\hat{\mu}$ as a normal distribution
$p(x) = N(x\mid \hat{\mu}, \hat{\Sigma})$ that approximates the posterior distribution with density $p$ such that

$$
F(p \mid q) = \int p(x) \cdot \lVert \nabla p(x) - \nabla q(x)\rVert_{\hat{\Sigma}}^2 dx
$$

is small. (Where $\lVert x\rVert_{\hat{\Sigma}}$ is the norm defined by the
preconditioner). Equivalently as an affine transformation
$T(x) = \hat{\Sigma}^\tfrac{1}{2}x + \mu$
such that $F(p, T) = \int p(x) \cdot \lVert\nabla T(x) -
\nabla N(x\mid 0, I)\rVert ^ 2 dx$ is minimal.

Given an arbitrary but sufficiently nice posterior $p$, this is minimal if
$\hat{\Sigma}$ is the geodesic mean of the covariance of $p$ and the inverse
of the covariance of $\nabla p$ (TODO double check). If $p$ is normal, then $Cov(\nabla p) = Cov(p)^{-1}$, so the minimum is reached at the covariance matrix.

If we only allow diagonal preconditioning matrices, we can find the minimum
analytically as 
$$
C = \text{diag}\left(\sqrt{\frac{\text{Var}(p)}{\text{Var}(\nabla p)}}\right).
$$

This diagonal preconditioner is already implemented in pymc and nuts-rs.

If we approximate the integral in $F$ with a finite number of samples using a monte carlo estimete, we find that $F$ is minimal if
$$
\text{Cov}(x_i) = \hat{\Sigma} \text{Cov}(\nabla x_i) \hat{\Sigma}
$$

If we have more dimensions than draws this does not have a unique solution,
so we introduce regularization. Some regularization methods based on the logdet or trace of $\Sigma$ or $\Sigma^{-1}$ still allow more or less explicit solutions as a algebraic Riccaati equations that sometimes can be made to scale reasonably with
the dimension, but in my experiments the geodesic distance to $I$, $R(\hat\Sigma)=\sum\log(\sigma_i) ^ 2$ seems to work better.

To avoid quadratic memory and computational costs with the dimensionality,
we write $\hat\Sigma = D(I + Q\Sigma Q^T - QQ^T)D$ where $Q\in\mathbb{R}^{N\times k}$ orthogonal and $D, \Sigma$ diagonal, so that we can perform
all operations necessary for hmc or nuts in $O(Nk^2)$.

We can now define a riemmannian metric on the space of all $(D, Q, \Sigma)$
as a pullback of the fisher information metric of $N(0, \hat\Sigma)$
and minimize $F$ using natural gradient descent. If we do this during tuning, we get similar behaviour as in a stochastic natural descent, and
can avoid the saddle points during optimization.

## Acknowledgment

A lot of the work that went into this package was during my time at Quantopian,
while trying to improve sampling of a (pretty awesome) model for portfolio
optimization. Thanks a lot for making that possible!

![Quantopian logo](https://raw.githubusercontent.com/pymc-devs/pymc3/master/docs/quantopianlogo.jpg)
