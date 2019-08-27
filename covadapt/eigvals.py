import numba
import numpy as np
from scipy import linalg, optimize


@numba.njit
def softplus(x):
    return np.log1p(np.exp(-x))


@numba.njit
def expit(x):
    return 1 / (1 + np.exp(-x))


@numba.njit
def soft_l1(v, alpha):
    return (np.logaddexp(0, alpha * v) + np.logaddexp(0, -alpha * v)).sum() / alpha


@numba.njit
def soft_l1_grad(v, alpha):
    return (expit(alpha * v) - expit(-alpha * v))


@numba.njit
def soft_l1_hessep(v, x, alpha):
    return 2 * alpha * x / (2 + np.exp(alpha * v) + np.exp(-alpha * v))


#@numba.njit
def cost(v, samples, gamma, alpha):
    k = samples.shape[0]

    #x_neg, x_pos = v[0], v[1]

    #A_xpos = samples.T @ (samples @ x_pos)
    #A_xneg = samples.T @ (samples @ x_neg)

    #v = x_pos - x_neg
    #variance = (v @ A_xpos - v @ A_xneg) / k

    #loss = gamma * np.linalg.norm(v, 1)

    #cost = -variance + loss
    #cost = -variance

    #var_grad_pos = 2 * (A_xpos - A_xneg)
    #var_grad_neg = 2 * (A_xneg - A_xpos)
    
    #return cost, -(var_grad_pos + var_grad_neg)
    
    tmp = np.dot(samples, v)
    tmp2 = np.dot(samples.T, tmp)
    variance = np.dot(v, tmp2) / k
    
    cost_ = -variance + gamma * np.linalg.norm(v, 1)
    #cost_ = -variance + gamma * soft_l1(v, alpha)
    
    grad = -2 * tmp2 / k + gamma * np.sign(v)
    #grad = -2 * tmp2 / k + gamma * soft_l1_grad(v, alpha)
    
    return cost_, grad
    #return cost_

#assert (
#    optimize.check_grad(
#        lambda x: cost(x, samples, 0.5)[0],
#        lambda x: cost(x, samples, 0.5)[1],
#        np.random.randn(n),
#    )
#    < 1e-3
#)

#@numba.njit
def hessian_p(v, p, samples, gamma, alpha):
    k = samples.shape[0]
    tmp = np.dot(samples, p)
    tmp2 = np.dot(samples.T, tmp) / k
    return -2 * tmp2
    #return -2 * tmp2 + gamma * soft_l1_hessep(v, p, alpha)


#def cost(v, A, gamma):
#    return -v @ A @ v + gamma * np.linalg.norm(v, 1)

@numba.njit
def cons_f(x):
    return np.array([np.linalg.norm(x)**2 - 1])

@numba.njit
def cons_fjac(x):
    return 2*x

nonlinear_constraint = optimize.NonlinearConstraint(cons_f, 0, 0, jac=cons_fjac)

def find_eigvec(samples, gamma, orthogonal_to=None, alpha=1e6):
    constraints = [nonlinear_constraint]
    if orthogonal_to is not None and len(orthogonal_to) > 0:
        ortho = optimize.LinearConstraint(orthogonal_to, 0, 0)
        constraints.append(ortho)
    
    U, _, _ = linalg.svd(samples.T, full_matrices=False)
    x = U[:, 0]
    #x_neg = np.maximum(0, -x)
    #x_pos = np.maximum(0, x)
    #x = np.stack([x_neg, x_pos])
    #x = np.random.randn(samples.shape[1])
    opt = optimize.minimize(
        cost,
        x,
        method='trust-constr',
        constraints=constraints,
        args=(samples, gamma, alpha),
        jac=True,
        #hessp=hessian_p,
        options={'maxiter': 100000}
    )
    #print(opt)
    assert opt.success, opt.message
    vec = opt.x / linalg.norm(opt.x)
    k = samples.shape[0]
    return vec, vec @ (samples.T @ (samples @ vec)) / k


def remove_axis(samples, vec):
    return samples - np.dot(samples, vec)[:, None] * vec[None, :]


def eigh_regularized_grad(samples, grad_samples, n_eigs, n_eigs_grad, gamma, gamma_grad, alpha=1e5):
    vecs = []
    vals = []
    samples_ = samples.copy()
    grad_samples_ = grad_samples.copy()

    for _ in range(n_eigs):
        vec, val = find_eigvec(samples_, gamma, np.array(vecs), alpha)
        samples_ = remove_axis(samples_, vec)
        grad_samples_ = remove_axis(grad_samples_, vec)
        vecs.append(vec)
        vals.append(val)
    
    for _ in range(n_eigs_grad):
        vec, val = find_eigvec(grad_samples_, gamma_grad, np.array(vecs), alpha)
        val = 1 / val
        samples_ = remove_axis(samples_, vec)
        grad_samples_ = remove_axis(grad_samples_, vec)
        vecs.append(vec)
        vals.append(val)
    
    vecs = np.array(vecs).T
    vals = np.array(vals)
    return vals, vecs
