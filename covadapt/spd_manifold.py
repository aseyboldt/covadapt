from copy import deepcopy
import time

import numpy as np
import scipy
from scipy import linalg
import pymanopt
import pymanopt.manifolds
import pymanopt.solvers
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.solvers.conjugate_gradient import BetaTypes
import sklearn
import torch
import aesara
import pymc as pm
import pandas as pd


class ExpStiefel(pymanopt.manifolds.Stiefel):
    def retr(self, X, G):
        # See eg http://arxiv.org/abs/2003.10374
        U = self.proj(X, G)

        W = linalg.expm(
            np.block([[X.T.dot(U), -U.T.dot(U)], [np.eye(self._p), X.T.dot(U)]])
        )
        Z = np.block([[linalg.expm(-X.T.dot(U))], [np.zeros((self._p, self._p))]])
        Y = np.block([X, U]).dot(W).dot(Z)
        return Y


class ScaledStiefelSPD(pymanopt.manifolds.manifold.Manifold):
    def __init__(
        self,
        n,
        k,
        verbosity=0,
        stiefel_retraction=True,
        exp_stiefel=True,
        enforce_no_signchange=False,
    ):
        self._log_diag = pymanopt.manifolds.Euclidean(n)

        self._log_vals = pymanopt.manifolds.Euclidean(k)
        if exp_stiefel:
            self._vecs = ExpStiefel(n, k)
        else:
            self._vecs = pymanopt.manifolds.Stiefel(n, k)

        self.verbosity = verbosity
        self._stiefel_retraction = stiefel_retraction

        self._n, self._k = n, k
        self._enforce_no_signchange = enforce_no_signchange

        self._manifolds = (
            self._log_diag,
            self._log_vals,
            self._vecs,
        )
        name = "Scaled Stiefel SPD manifold: {:s}".format(
            " x ".join([str(man) for man in self._manifolds])
        )
        dimension = np.sum([man.dim for man in self._manifolds])
        point_layout = tuple(manifold.point_layout for manifold in self._manifolds)
        super().__init__(name, dimension, point_layout=point_layout)

    def inner(self, X, G, H):
        vals = np.exp(X[1])

        U = X[2]

        dd1, dl1, dU1 = G
        dd2, dl2, dU2 = H

        dA1 = U.T @ dU1
        dA2 = U.T @ dU2

        ivals = 1 / vals
        vals_bar = vals - 1
        ivals_bar = ivals - 1

        i_uu = ((dU1 * dU2) @ (vals_bar**2 * ivals)).sum() + ivals_bar @ (
            dA1 * dA2
        ) @ vals_bar

        i_dd = (
            2 * dd1 @ dd2
            + (dd1 * dd2) @ (U * U) @ (ivals + vals - 2)
            + ivals_bar
            @ ((U.T @ (dd1[:, None] * U)) * (U.T @ (dd2[:, None] * U)))
            @ vals_bar
        )

        i_ud = (
            dd2 @ (dU1 * U) @ ((ivals + 1) * vals_bar)
            + (
                (
                    U
                    @ (
                        (
                            ivals[:, None] * vals[None, :]
                            + vals[:, None]
                            - ivals[:, None]
                        )
                        * dA1
                        + dA1
                    )
                )
                * U
            ).sum(1)
            @ dd2
        )

        i_du = (
            dd1 @ (dU2 * U) @ ((ivals + 1) * vals_bar)
            + (
                (
                    U
                    @ (
                        (
                            ivals[:, None] * vals[None, :]
                            + vals[:, None]
                            - ivals[:, None]
                        )
                        * dA2
                        + dA2
                    )
                )
                * U
            ).sum(1)
            @ dd1
        )

        i_ll = dl1 @ dl2 / 2

        i_ld = dd1 @ (U * U) @ dl2 + dd2 @ (U * U) @ dl1

        if self.verbosity >= 3:
            print("Inner product")
            print("UU:", i_uu)
            print("dd", i_dd)
            print("ll", i_ll)
            print("Ud", i_ud)
            print("dU", i_du)
            print("ld", i_ld)
            print("Current log(lamda)", np.log(vals))

        return i_uu + i_dd + i_ud + i_du + i_ll + i_ld

    def egrad2rgrad(self, X, dx):
        d, l, U = X

        N, K = U.shape

        errstate = {}
        try:
            errstate = np.seterr(all="raise")
            d = np.exp(d)
            l = np.exp(l)
        finally:
            np.seterr(**errstate)

        dd, dl, dU = dx

        def woodbury(D, U, C, V, x):
            D_inv = 1 / D
            Dinv_x = D_inv * x
            VDinv_x = V @ Dinv_x

            C_inv = 1 / C
            inner = np.diag(C_inv) + (V * D_inv) @ U

            return Dinv_x - D_inv * (U @ linalg.solve(inner, VDinv_x))

        lp = l[:, None]
        lq = l[None, :]

        hA = U.T @ dU

        scale = lp - lq
        scale[np.diag_indices_from(scale)] = 1  # avoid warnings in devision
        scale = lq / scale
        scale[np.diag_indices_from(scale)] = 0

        lhs = (
            dd
            - 2 * U**2 @ dl
            + (
                scale[None, :, :]
                * U[:, :, None]
                * U[:, None, :]
                * (hA - hA.T)[None, :, :]
            )
            .sum(-1)
            .sum(-1)
            + (((dU.T @ U) @ U.T) * (U * ((l + 1) / (l - 1))).T).sum(0)
            - (((l + 1) / (l - 1))[None, :] * U * dU).sum(-1)
        )

        V = (U[:, None, :] * U[:, :, None]).reshape((N, K * K))
        lp = l[:, None]
        lq = l[None, :]
        L = lp - lq + 2
        inner = L.reshape(-1)

        diag = 2 * np.ones(N) - 4 * (U**2).sum(1)
        grad_d = woodbury(diag, V, inner, V.T, lhs)

        grad_l = 2 * (dl - np.diag(U.T @ (grad_d[:, None] * U)))

        L = (l[None, :] - l[:, None]) ** 2
        L[np.diag_indices_from(L)] = 1  # avoid warnings

        g_a = (
            l[None, :] * l[:, None] * (U.T @ dU - dU.T @ U)
            + np.diag(l**2) @ U.T @ (grad_d[:, None] * U)
            - (U.T @ (grad_d[:, None] * U)) * l**2
        ) / L
        g_a[np.diag_indices_from(g_a)] = 0

        du_scaled = dU * (l / (l - 1) ** 2)
        u_scaled = (U * ((l + 1) / (l - 1))) * grad_d[:, None]
        Ub_gB = du_scaled - U @ (U.T @ du_scaled) - u_scaled + U @ (U.T @ u_scaled)

        grad_U = U @ g_a + Ub_gB

        grad = [grad_d, grad_l, grad_U]

        check = False
        if check:
            dz = [np.random.randn(*val.shape) for val in dx]
            dz[2] = self._vecs.randvec(U)

            inner = self.inner(X, grad, dz, individual=True)
            directional = (
                np.inner(dd, dz[0]) + np.inner(dl, dz[1]) + np.trace(dU.T @ dz[2])
            )
            np.testing.assert_allclose(inner, directional)

        return _ProductTangentVector([g for g in grad])

    def perturb(self, X, amount):
        _, _, U = X
        n, k = U.shape
        grads = [
            amount * np.random.randn(n),
            amount * np.random.randn(k),
            amount * np.random.randn(n, k),
        ]
        grads = self.proj(X, grads)
        return self.retr(X, grads)

    def retr(self, X, dx):
        if self._stiefel_retraction:
            diag, vals, U = [
                man.retr(X[k], dx[k]) for k, man in enumerate(self._manifolds)
            ]

            if not self._enforce_no_signchange:
                return [diag, vals, U]

            l_orig = X[1]
            is_pos = l_orig >= 0

            vals[is_pos] = np.where(vals[is_pos] >= 0, vals[is_pos], l_orig[is_pos])
            vals[~is_pos] = np.where(vals[~is_pos] < 0, vals[~is_pos], l_orig[~is_pos])

            assert (vals[is_pos] >= 0).all()
            assert (vals[~is_pos] < 0).all()

            return [diag, vals, U]

        diag, vals, U = X

        dd, dl, dU = dx

        dA = U.T @ dU
        dU = dU - 0.5 * U @ (dA + dA.T)

        V = (U + dU) * np.exp((vals + dl) / 2)[None, :]
        U_out, l_out, _ = linalg.svd(V, full_matrices=False)
        return [diag + dd, 2 * np.log(l_out), U_out]

    ## From Product
    def __setattr__(self, key, value):
        if hasattr(self, key):
            if key == "manifolds":
                raise AttributeError("Cannot override 'manifolds' attribute")
        super().__setattr__(key, value)

    @property
    def typicaldist(self):
        return np.sqrt(np.sum([man.typicaldist**2 for man in self._manifolds]))

    def norm(self, X, G):
        return np.sqrt(np.clip(self.inner(X, G, G), 1e-10, np.inf))

    def proj(self, X, U):
        return _ProductTangentVector(
            [man.proj(X[k], U[k]) for k, man in enumerate(self._manifolds)]
        )

    def rand(self):
        return [man.rand() for man in self._manifolds]

    def randvec(self, X):
        scale = len(self._manifolds) ** (-1 / 2)
        return _ProductTangentVector(
            [scale * man.randvec(X[k]) for k, man in enumerate(self._manifolds)]
        )

    def zerovec(self, X):
        return _ProductTangentVector(
            [man.zerovec(X[k]) for k, man in enumerate(self._manifolds)]
        )

    def transp(self, X1, X2, G):
        return _ProductTangentVector(
            [man.transp(X1[k], X2[k], G[k]) for k, man in enumerate(self._manifolds)]
        )


def _make_problem(
    draws,
    grads,
    *,
    n_eigs,
    gamma,
    alpha,
    delta,
    beta,
    estimate_diag=True,
    use_scaled_stiefel_spd=True,
    verbosity=0,
    stiefel_retraction=True,
    exp_stiefel=True,
    use_grads=True,
):
    draw_std = np.std(draws, 0)
    grad_std = np.std(grads, 0)

    _, n_dim = draws.shape

    if estimate_diag:
        stds_guess = np.sqrt(draw_std / grad_std)
        stds = np.ones_like(stds_guess)
    else:
        stds = np.sqrt(draw_std / grad_std)
        stds_guess = stds

    x_hat = (draws - draws.mean(0, keepdims=True)) / stds[None, :]
    p_hat = (grads - grads.mean(0, keepdims=True)) * stds[None, :]

    x_hat_torch = torch.as_tensor(x_hat)
    p_hat_torch = torch.as_tensor(p_hat)

    vecs = ExpStiefel(n_dim, n_eigs)
    log_vals = pymanopt.manifolds.Euclidean(n_eigs)

    if estimate_diag:
        log_diag = pymanopt.manifolds.Euclidean(n_dim)
        manifold = pymanopt.manifolds.Product([vecs, log_vals, log_diag])
    else:
        manifold = pymanopt.manifolds.Product([vecs, log_vals])

    if use_scaled_stiefel_spd:
        manifold = ScaledStiefelSPD(
            n_dim,
            n_eigs,
            verbosity=verbosity,
            stiefel_retraction=stiefel_retraction,
            exp_stiefel=exp_stiefel,
            enforce_no_signchange=True,
        )

    @pymanopt.function.pytorch(manifold)
    def cost(*args):
        if estimate_diag:
            vecs, log_vals, log_diag = args
        else:
            vecs, log_vals = args

        if use_scaled_stiefel_spd:
            log_diag, log_vals, vecs = args

        val_mean = 0

        reg_vals = gamma * ((log_vals - val_mean) ** 2).sum()
        reg_vecs = -(
            delta * torch.special.logsumexp((alpha - 1) * torch.log(vecs**2), 0).sum()
        )

        if estimate_diag:
            diag = torch.exp(log_diag)
            x_hat_scaled = x_hat_torch / diag[None, :]
            p_hat_scaled = p_hat_torch * diag[None, :]
            reg_diag = beta * (log_diag**2).sum()
        else:
            x_hat_scaled = x_hat_torch
            p_hat_scaled = p_hat_torch
            reg_diag = 0

        others = torch.exp(torch.as_tensor(val_mean))

        vals = torch.expm1(log_vals)
        sigma_prod = (vecs * vals) @ (vecs.T @ p_hat_scaled.T) + p_hat_scaled.T * others

        vals_inv = torch.expm1(-log_vals)
        omega_prod = (vecs * vals_inv) @ (
            vecs.T @ x_hat_scaled.T
        ) + x_hat_scaled.T / others

        loss_p = (p_hat_scaled * sigma_prod.T).sum()
        loss_x = (x_hat_scaled * omega_prod.T).sum()

        if use_grads:
            loss = loss_x + loss_p
        else:
            loss = 2 * loss_x

        return loss + reg_diag + reg_vals + reg_vecs

    return pymanopt.Problem(manifold, cost, verbosity=verbosity)


class ManifoldSolver:
    def __init__(
        self,
        *,
        use_scaled_stiefel_spd=True,
        n_eigs=2,
        gamma=0.1,
        beta=0.0,
        alpha=0.0,
        delta=0.0,
        use_grads=True,
        solver_method="steepest-descent",
        exp_stiefel=False,
        stiefel_retraction=True,
        optimizer_args=None,
    ):
        self.estimate_diag = True
        self.n_eigs = n_eigs
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.use_grads = use_grads
        self._last_solve_dims = None
        self._use_scaled_stiefel_spd = use_scaled_stiefel_spd
        self.solver_method = solver_method
        self.exp_stiefel = exp_stiefel
        self.stiefel_retraction = stiefel_retraction
        if optimizer_args is None:
            optimizer_args = {}
        self.optimizer_args = optimizer_args

    def get_params(self, deep=True):
        params = {}
        for name in [
            "n_eigs",
            "gamma",
            "beta",
            "alpha",
            "delta",
            "optimizer_args",
            "use_grads",
            "solver_method",
            "exp_stiefel",
            "stiefel_retraction",
        ]:
            params[name] = getattr(self, name)
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X_draws, X_grads, *, reuse_old_solution=True):
        self.X_draws_ = X_draws
        self.X_grads_ = X_grads
        self.n_draws_, self.n_dim_ = self.X_draws_.shape

        draw_std = np.std(X_draws, axis=0)
        grad_std = np.std(X_grads, axis=0)

        draw_std = np.clip(draw_std, 1e-6, 1e6)
        grad_std = np.clip(grad_std, 1e-6, 1e6)

        problem = _make_problem(
            X_draws,
            X_grads,
            n_eigs=self.n_eigs,
            alpha=self.alpha,
            beta=self.beta,
            delta=self.delta,
            gamma=self.gamma,
            estimate_diag=self.estimate_diag,
            use_grads=self.use_grads,
            use_scaled_stiefel_spd=self._use_scaled_stiefel_spd,
            exp_stiefel=self.exp_stiefel,
            stiefel_retraction=self.stiefel_retraction,
            verbosity=self.optimizer_args.get("verbosity", 0),
        )

        args = self.optimizer_args.copy()
        if "verbosity" in args:
            del args["verbosity"]
        if self.solver_method == "trust-region":
            solver = pymanopt.solvers.TrustRegions(**args)
        elif self.solver_method == "steepest-descent":
            solver = pymanopt.solvers.SteepestDescent(**args)
        elif self.solver_method == "cg":
            solver = pymanopt.solvers.ConjugateGradient(**args)
        elif isinstance(self.solver_method, pymanopt.solvers.solver.Solver):
            solver = self.solver_method
        else:
            raise ValueError(f"Unknown solver method: {self.solver_method}")

        if (
            reuse_old_solution
            and self._last_solve_dims is not None
            and self._last_solve_dims == (self.n_dim_, self.n_eigs, self.estimate_diag)
        ):
            start = self._last_solution
            if self.optimizer_args.get("verbosity", 0) >= 1:
                print("reusing...", np.array(list(sorted(start[1]))))
        else:
            start_vecs = np.random.randn(self.n_dim_, self.n_eigs)
            start_vecs, _ = linalg.qr(start_vecs, mode="economic")

            start_vals = np.random.randn(self.n_eigs) * 0.1
            sign = (np.arange(self.n_eigs) < self.n_eigs // 2) - 0.5

            start_vals = np.copysign(start_vals, sign)
            self._last_start_vals = start_vals

            if self.estimate_diag:
                stds_guess = np.sqrt(X_draws.std(0) / X_grads.std(0))
                start = [start_vecs, start_vals, np.log(stds_guess)]
            else:
                start = [start_vecs, start_vals]

            if self._use_scaled_stiefel_spd:
                start = [start[2], start[1], start[0]]

        if self.optimizer_args.get("verbosity", 0) > 2:
            print("start:")
            print(start[0])
            print(start[1])
            print(start[2])

        solution = solver.solve(problem, x=start)
        final_cost = problem.cost(solution)

        self._last_solution = solution
        self._last_solve_dims = (self.n_dim_, self.n_eigs, self.estimate_diag)

        if self._use_scaled_stiefel_spd:
            solution = [solution[2], solution[1], solution[0]]
        else:
            solution = [solution[0], solution[1], solution[2]]

        self.vecs_, self.vals_, self.stds_ = solution
        self.vals_ = np.exp(self.vals_)
        self.stds_ = np.exp(self.stds_)
        self.final_cost_ = final_cost

        return self


class CovarianceEstimator(sklearn.base.BaseEstimator):
    def __init__(self, estimator, *, compute_full_matrix="auto", logeigval_cutoff=0.05):
        super().__init__()
        self.compute_full_matrix = compute_full_matrix
        self.estimator = estimator
        self._param_names = list(self.estimator.get_params())
        self.logeigval_cutoff = logeigval_cutoff

    def get_params(self, deep=True, *, estimator_only=False):
        params = {}
        params.update(
            {
                "compute_full_matrix": self.compute_full_matrix,
                "logeigval_cutoff": self.logeigval_cutoff,
                "estimator": self.estimator,
            }
        )
        return params

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            if parameter in self._param_names:
                self.estimator.set_params(**{parameter: value})
            else:
                setattr(self, parameter, value)
        return self

    def _more_tags(self):
        return {
            "X_types": ["2darray", "2darray"],
        }

    def fit(self, X_draws, X_grads):
        X_draws = sklearn.base.check_array(X_draws)
        X_grads = sklearn.base.check_array(X_grads)

        self.X_draws_ = X_draws
        self.X_grads_ = X_grads
        self.n_draws_, self.n_dim_ = self.X_draws_.shape

        (
            self.estimator.set_params(**self.get_params(estimator_only=True)).fit(
                X_draws,
                X_grads,
            )
        )
        vals = self.estimator.vals_
        vecs = self.estimator.vecs_
        others = getattr(self.estimator, "others_", 1.0)
        stds = self.estimator.stds_

        within_cutoff = np.abs(np.log(vals) - np.log(others)) < self.logeigval_cutoff
        self.eigvals_ = vals[~within_cutoff]
        self.eigvecs_ = vecs[:, ~within_cutoff].copy(order="F")
        self.other_eigvals_ = others
        self.stds_ = stds

        if not np.isfinite(stds).all():
            raise linalg.LinAlgError("Bad diagonal approximation")
        if not np.isfinite(vals).all() and (vals > 0).all():
            raise linalg.LinAlgError("Bad eigenvalue approximation")

        if self.compute_full_matrix == "auto":
            compute_full_matrix = self.n_dim_ < 50
        else:
            compute_full_matrix = self.compute_full_matrix

        if compute_full_matrix:
            self.full_matrix_computed_ = True
            self.covariance_ = vecs @ np.diag(vals - others) @ vecs.T + others * np.eye(
                self.n_dim_
            )
            self.covariance_[...] *= stds[None, :]
            self.covariance_[...] *= stds[:, None]
            self.precision_ = vecs @ np.diag(1 / vals - 1 / others) @ vecs.T + (
                1 / others
            ) * np.eye(self.n_dim_)
            self.precision_[...] /= stds[None, :]
            self.precision_[...] /= stds[:, None]
        else:
            self.full_matrix_computed_ = False

        self.mean_ = self.matmul(X_grads.mean(0)) + X_draws.mean(0)

        return self

    def score(self, X_draws, X_grads):
        sklearn.utils.validation.check_is_fitted(self)

        # Input validation
        X_draws = sklearn.base.check_array(X_draws)
        X_grads = sklearn.base.check_array(X_grads)

        expected_grads = -self.inv_matmul((X_draws - self.mean_).T).T
        diff = X_grads - expected_grads
        return -(diff * self.matmul(diff.T).T).sum(-1).mean()

    def matmul(self, X):
        sklearn.utils.validation.check_is_fitted(self)

        if self.full_matrix_computed_:
            return self.covariance_ @ X

        vals, vecs = self.eigvals_, self.eigvecs_
        stds = self.stds_
        others = self.other_eigvals_

        return (matmul_eigs(vals, vecs, others, (X.T * stds).T).T * stds).T

    def matmul_single(self, x):
        sklearn.utils.validation.check_is_fitted(self)

        if self.full_matrix_computed_:
            return self.covariance_ @ x

        vals, vecs = self.eigvals_, self.eigvecs_
        stds = self.stds_
        others = self.other_eigvals_

        return matmul_eigs_single(vals, vecs, others, x * stds) * stds

    def inv_matmul(self, X):
        sklearn.utils.validation.check_is_fitted(self)

        if self.full_matrix_computed_:
            return self.precision_ @ X

        vals, vecs = self.eigvals_, self.eigvecs_
        stds = self.stds_
        others = self.other_eigvals_

        return (matmul_eigs(1 / vals, vecs, 1 / others, (X.T / stds).T).T / stds).T

    def draw(self):
        sklearn.utils.validation.check_is_fitted(self)

        x = np.random.randn(self.n_dim_)

        vals, vecs = self.eigvals_, self.eigvecs_
        stds = self.stds_
        others = self.other_eigvals_

        return matmul_eigs(np.sqrt(vals), vecs, np.sqrt(others), x) * stds

    def draw_inv(self):
        sklearn.utils.validation.check_is_fitted(self)

        x = np.random.randn(self.n_dim_)

        vals, vecs = self.eigvals_, self.eigvecs_
        stds = self.stds_
        others = self.other_eigvals_

        return (
            matmul_eigs_single(1 / np.sqrt(vals), vecs, 1 / np.sqrt(others), x) / stds
        )


def matmul_eigs(vals, vecs, others, x):
    x1 = vecs.T @ x
    x1 = ((vals - others) * x1.T).T
    x1 = vecs @ x1
    return x * others + x1


gemv = linalg.blas.get_blas_funcs("gemv")
axpy = linalg.blas.get_blas_funcs("axpy")


def matmul_eigs_single(vals, vecs, others, x):
    x_inner = gemv(1.0, vecs, x, trans=1)
    x_inner = (vals - others) * x_inner
    out = gemv(1.0, vecs, x_inner)
    axpy(x, out, a=others)
    return out


class SteepestDescent(pymanopt.solvers.solver.Solver):
    """Riemannian steepest descent solver.
    Perform optimization using gradient descent with line search.
    This method first computes the gradient of the objective, and then
    optimizes by moving in the direction of steepest descent (which is the
    opposite direction to the gradient).
    Args:
        linesearch: The line search method.
    """

    def __init__(
        self,
        linesearch=None,
        perturbance_cooldown=50,
        perturbance_threshold=1e-1,
        perturbance_stepsize_threshold=1e-5,
        perturbance=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._perturbance_cooldown = perturbance_cooldown
        self._perturbance_threshold = perturbance_threshold
        self._perturbance_stepsize_threshold = perturbance_stepsize_threshold
        self._perturbance = perturbance

        if linesearch is None:
            self._linesearch = pymanopt.solvers.linesearch.LineSearchBackTracking()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    # Function to solve optimisation problem using steepest descent.
    def solve(self, problem, x=None, reuselinesearch=False):
        """Run steepest descent algorithm.
        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            x: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuselinesearch: Whether to reuse the previous linesearch object.
                Allows to use information from a previous call to
                :meth:`solve`.
        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        perturbance_iter = self._perturbance_cooldown
        perturbance_cooldown = self._perturbance_cooldown

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            iter_format_length = int(np.log10(self._maxiter)) + 1
            column_printer = pymanopt.tools.printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iter_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                    ("Stepsize", ".8e"),
                ]
            )
        else:
            column_printer = pymanopt.tools.printer.VoidPrinter()

        column_printer.print_header()

        self._start_optlog(
            extraiterfields=["gradnorm"],
            solverparams={"linesearcher": linesearch},
        )

        # Initialize iteration counter and timer
        iter = 0
        time0 = time.time()

        stepsize = np.nan
        perturbance = self._perturbance

        while True:
            # Calculate new cost, grad and gradnorm
            cost = objective(x)
            grad = gradient(x)
            gradnorm = man.norm(x, grad)
            iter = iter + 1
            perturbance_iter += 1

            column_printer.print_row([iter, cost, gradnorm, stepsize])

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            # Descent direction is minus the gradient
            desc_dir = -grad

            perturbing = perturbance_iter > perturbance_cooldown and (
                # gradnorm < self._perturbance_threshold
                # or
                stepsize
                < self._perturbance_stepsize_threshold
            )

            if perturbing:
                local_perturbance = perturbance
                while True:
                    newx = man.perturb(x, perturbance)
                    newcost = objective(newx)
                    newgrad = gradient(newx)
                    if np.isfinite(newcost) and all(
                        np.isfinite(val).all() for val in newgrad
                    ):
                        break

                    if verbosity >= 2:
                        print(
                            "Conjugate gradient info: cost or grad not finite after perturbance."
                        )

                    local_perturbance /= 2

                if verbosity >= 2:
                    print(f"SteepestDescent: perturbing with perturbance {perturbance}")

                x = newx
                perturbance_iter = 0
                # perturbance_cooldown *= 2
                perturbance /= 2
                stepsize = np.nan
                linesearch = deepcopy(self._linesearch)
                self.linesearch = linesearch
            else:
                # Perform line-search
                stepsize, x = linesearch.search(
                    objective, man, x, desc_dir, cost, -(gradnorm**2)
                )

                stop_reason = self._check_stopping_criterion(
                    time0, stepsize=stepsize, gradnorm=gradnorm, iter=iter
                )

                if stop_reason:
                    if verbosity >= 1:
                        print(stop_reason)
                        print("")
                    break

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(
                x,
                objective(x),
                stop_reason,
                time0,
                stepsize=stepsize,
                gradnorm=gradnorm,
                iter=iter,
            )
            return x, self._optlog


# Adapted from pymanopt implementation
class PerturbingConjugateGradient(pymanopt.solvers.solver.Solver):
    """Riemannian conjugate gradient method.

    Perform optimization using nonlinear conjugate gradient method with
    linesearch.
    This method first computes the gradient of the cost function, and then
    optimizes by moving in a direction that is conjugate to all previous search
    directions.

    Args:
        beta_type: Conjugate gradient beta rule used to construct the new
            search direction.
        orth_value: Parameter for Powell's restart strategy.
            An infinite value disables this strategy.
            See in code formula for the specific criterion used.
        linesearch: The line search method.
    """

    def __init__(
        self,
        beta_type=pymanopt.solvers.conjugate_gradient.BetaTypes.HestenesStiefel,
        orth_value=np.inf,
        linesearch=None,
        perturbance_cooldown=50,
        perturbance_threshold=1e-1,
        perturbance_stepsize_threshold=1e-5,
        perturbance=0.01,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self._beta_type = beta_type
        self._orth_value = orth_value

        self._perturbance_cooldown = perturbance_cooldown
        self._perturbance_threshold = perturbance_threshold
        self._perturbance_stepsize_threshold = perturbance_stepsize_threshold
        self._perturbance = perturbance

        if linesearch is None:
            self._linesearch = pymanopt.solvers.linesearch.LineSearchAdaptive()
        else:
            self._linesearch = linesearch
        self.linesearch = None

    def solve(self, problem, x=None, reuselinesearch=False):
        """Run CG method.

        Args:
            problem: Pymanopt problem class instance exposing the cost function
                and the manifold to optimize over.
                The class must either
            x: Initial point on the manifold.
                If no value is provided then a starting point will be randomly
                generated.
            reuselinesearch: Whether to reuse the previous linesearch object.
                Allows to use information from a previous call to
                :meth:`solve`.

        Returns:
            Local minimum of the cost function, or the most recent iterate if
            algorithm terminated before convergence.
        """
        man = problem.manifold
        verbosity = problem.verbosity
        objective = problem.cost
        gradient = problem.grad

        perturbance_iter = self._perturbance_cooldown
        perturbance_cooldown = self._perturbance_cooldown

        if not reuselinesearch or self.linesearch is None:
            self.linesearch = deepcopy(self._linesearch)
        linesearch = self.linesearch

        # If no starting point is specified, generate one at random.
        if x is None:
            x = man.rand()

        if verbosity >= 1:
            print("Optimizing...")
        if verbosity >= 2:
            iter_format_length = int(np.log10(self._maxiter)) + 1
            column_printer = pymanopt.tools.printer.ColumnPrinter(
                columns=[
                    ("Iteration", f"{iter_format_length}d"),
                    ("Cost", "+.16e"),
                    ("Gradient norm", ".8e"),
                    ("Stepsize", ".8e"),
                ]
            )
        else:
            column_printer = pymanopt.tools.printer.VoidPrinter()

        column_printer.print_header()

        # Calculate initial cost-related quantities
        cost = objective(x)
        grad = gradient(x)
        gradnorm = man.norm(x, grad)
        Pgrad = problem.precon(x, grad)
        gradPgrad = man.inner(x, grad, Pgrad)

        # Initial descent direction is the negative gradient
        desc_dir = -Pgrad

        self._start_optlog(
            extraiterfields=["gradnorm"],
            solverparams={
                "beta_type": self._beta_type,
                "orth_value": self._orth_value,
                "linesearcher": linesearch,
            },
        )

        # Initialize iteration counter and timer
        iter = 0
        stepsize = np.nan
        time0 = time.time()

        while True:
            column_printer.print_row([iter, cost, gradnorm, stepsize])

            if self._logverbosity >= 2:
                self._append_optlog(iter, x, cost, gradnorm=gradnorm)

            # The line search algorithms require the directional derivative of
            # the cost at the current point x along the search direction.
            df0 = man.inner(x, grad, desc_dir)

            # If we didn't get a descent direction: restart, i.e., switch to
            # the negative gradient. Equivalent to resetting the CG direction
            # to a steepest descent step, which discards the past information.
            if df0 >= 0:
                # Or we switch to the negative gradient direction.
                if verbosity >= 3:
                    print(
                        "Conjugate gradient info: got an ascent direction "
                        f"(df0 = {df0:.2f}), reset to the (preconditioned) "
                        "steepest descent direction."
                    )
                # Reset to negative gradient: this discards the CG memory.
                desc_dir = -Pgrad
                df0 = -gradPgrad

            perturbing = perturbance_iter > perturbance_cooldown and (
                gradnorm < self._perturbance_threshold
                or stepsize < self._perturbance_stepsize_threshold
            )
            if perturbing:
                if verbosity >= 2:
                    print(
                        "Conjugate gradient info: perturbing to avoid saddle points..."
                    )
                perturbance = self._perturbance
                while True:
                    newx = man.perturb(x, perturbance)
                    newcost = objective(newx)
                    newgrad = gradient(newx)
                    if np.isfinite(newcost) and all(
                        np.isfinite(val).all() for val in newgrad
                    ):
                        break

                    if verbosity >= 2:
                        print(
                            "Conjugate gradient info: cost or grad not finite after perturbance."
                        )

                    perturbance /= 2

                perturbance_iter = 0
                perturbance_cooldown *= 2
                stepsize = np.nan
                linesearch = deepcopy(self._linesearch)
                self.linesearch = linesearch

            else:
                stop_reason = self._check_stopping_criterion(
                    time0, gradnorm=gradnorm, iter=iter + 1, stepsize=stepsize
                )

                if stop_reason:
                    if verbosity >= 1:
                        print(stop_reason)
                        print("")
                    break

                # Execute line search
                stepsize, newx = linesearch.search(
                    objective, man, x, desc_dir, cost, df0
                )
                newcost = objective(newx)
                newgrad = gradient(newx)

                if not np.isfinite(newcost) or not all(
                    np.isfinite(val).all() for val in newgrad
                ):
                    if verbosity >= 2:
                        print(
                            "Conjugate gradient info: cost or grad not finite after gradient step."
                        )

                    newx = x
                    stepsize = 0
                    newcost = cost
                    newgrad = df0

            # Compute the new cost-related quantities for newx
            # newcost = objective(newx)
            # newgrad = gradient(newx)
            newgradnorm = man.norm(newx, newgrad)
            Pnewgrad = problem.precon(newx, newgrad)
            newgradPnewgrad = man.inner(newx, newgrad, Pnewgrad)

            # Apply the CG scheme to compute the next search direction
            oldgrad = man.transp(x, newx, grad)
            orth_grads = man.inner(newx, oldgrad, Pnewgrad) / newgradPnewgrad

            # Powell's restart strategy (see page 12 of Hager and Zhang's
            # survey on conjugate gradient methods, for example)
            # print(orth_grads)
            if perturbing or abs(orth_grads) >= self._orth_value:
                beta = 0
                desc_dir = -Pnewgrad
            else:
                desc_dir = man.transp(x, newx, desc_dir)

                if self._beta_type == BetaTypes.FletcherReeves:
                    beta = newgradPnewgrad / gradPgrad
                elif self._beta_type == BetaTypes.PolakRibiere:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    beta = max(0, ip_diff / gradPgrad)
                elif self._beta_type == BetaTypes.HestenesStiefel:
                    diff = newgrad - oldgrad
                    ip_diff = man.inner(newx, Pnewgrad, diff)
                    try:
                        beta = max(0, ip_diff / man.inner(newx, diff, desc_dir))
                    # if ip_diff = man.inner(newx, diff, desc_dir) = 0
                    except ZeroDivisionError:
                        beta = 1
                elif self._beta_type == BetaTypes.HagerZhang:
                    diff = newgrad - oldgrad
                    Poldgrad = man.transp(x, newx, Pgrad)
                    Pdiff = Pnewgrad - Poldgrad
                    deno = man.inner(newx, diff, desc_dir)
                    numo = man.inner(newx, diff, Pnewgrad)
                    numo -= (
                        2
                        * man.inner(newx, diff, Pdiff)
                        * man.inner(newx, desc_dir, newgrad)
                        / deno
                    )
                    beta = numo / deno
                    # Robustness (see Hager-Zhang paper mentioned above)
                    desc_dir_norm = man.norm(newx, desc_dir)
                    eta_HZ = -1 / (desc_dir_norm * min(0.01, gradnorm))
                    beta = max(beta, eta_HZ)
                else:
                    types = ", ".join([f"BetaTypes.{t}" for t in BetaTypes._fields])
                    raise ValueError(
                        f"Unknown beta_type {self._beta_type}. Should be one "
                        f"of {types}."
                    )

                desc_dir = -Pnewgrad + beta * desc_dir

            # Update the necessary variables for the next iteration.
            x = newx
            cost = newcost
            grad = newgrad
            Pgrad = Pnewgrad
            gradnorm = newgradnorm
            gradPgrad = newgradPnewgrad

            iter += 1
            perturbance_iter += 1

        if self._logverbosity <= 0:
            return x
        else:
            self._stop_optlog(
                x,
                cost,
                stop_reason,
                time0,
                stepsize=stepsize,
                gradnorm=gradnorm,
                iter=iter,
            )
            return x, self._optlog


class QuadPotentialFullAdapt(pm.step_methods.hmc.quadpotential.QuadPotentialFull):
    """Adapt a dense mass matrix using the sample covariances."""

    def __init__(
        self,
        n,
        adaptation_window=50,
        adaptation_window_multiplier=1.1,
        update_window=20,
        adapt_stop=950,
        dtype=None,
        verbose=False,
        compute_full_matrix=False,
        n_eigs=8,
        gamma=0.1,
        alpha=0.5,
        beta=0,
        delta=0,
        maxiter=50,
        logeigval_cutoff=0.05,
    ):
        if dtype is None:
            dtype = aesara.config.floatX

        self._initial_cov = np.eye(n, dtype=dtype)
        self._initial_chol = scipy.linalg.cholesky(self._initial_cov, lower=True)

        self.dtype = dtype
        self._n = n
        self._n_eigs = n_eigs

        self._adaptation_window = int(adaptation_window)
        self._initial_adaptation_window = self._adaptation_window
        self._adaptation_window_multiplier = float(adaptation_window_multiplier)
        self._update_window = int(update_window)

        self._foreground = ([], [])
        self._background = ([], [])
        self._discard_window = 50
        self._adapt_stop = adapt_stop
        self.verbose = verbose
        self._compute_full_matrix = compute_full_matrix

        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._delta = delta
        self._maxiter = maxiter
        self._logeigval_cutoff = logeigval_cutoff

        self.reset()

    def reset(self):
        self._previous_update = self._discard_window
        self._cov = None
        self._tmp_cov = self._initial_cov.copy()
        self._tmp_chol = self._initial_chol.copy()
        self._foreground = ([], [])
        self._background = ([], [])
        self._n_samples = 0
        self._adaptation_window = self._initial_adaptation_window

        optimizer = PerturbingConjugateGradient(
            minstepsize=1e-6,
            mingradnorm=1e-2,
            perturbance_cooldown=20,
            perturbance=0.01,
            maxiter=self._maxiter,
            perturbance_stepsize_threshold=1e-4,
            # beta_type=pymanopt.solvers.conjugate_gradient.BetaTypes.PolakRibiere,
        )

        solver = ManifoldSolver(
            n_eigs=self._n_eigs,
            stiefel_retraction=True,
            exp_stiefel=True,
            optimizer_args={
                "verbosity": 0,
                "minstepsize": 1e-8,
                "mingradnorm": 1e-2,
                "maxiter": self._maxiter,
            },
            solver_method=optimizer,
            alpha=self._alpha,
            delta=self._delta,
            beta=self._beta,
            gamma=self._gamma,
        )

        estimator = CovarianceEstimator(solver, logeigval_cutoff=self._logeigval_cutoff)
        self._initial_estimator = estimator

    def _update_from_weightvar(self, samples, soft):
        draws, grads = [np.array(val) for val in samples]
        tmp = pd.DataFrame(draws)
        is_duplicate = tmp.duplicated().values
        draws = draws[~is_duplicate]
        grads = grads[~is_duplicate]

        if self.verbose:
            print("Unique: ", len(draws))

        try:
            if len(draws) > 1:
                if self.verbose:
                    print("Soft update")
                self._cov = self._initial_estimator
                self._cov.fit(*[np.array(val) for val in samples])
            else:
                scale = np.random.rand(len(grads[0]))
                self._initial_cov = np.diag(np.exp(np.log((grads**2).sum(0)) * scale))
                self._initial_chol = np.diag(np.sqrt(np.diag(self._initial_cov)))
            if self.verbose:
                if self._cov is not None:
                    print("num eigs:", len(self._cov.eigvals_))
                    print("eigvals:", list(sorted(self._cov.eigvals_)))
        except (ValueError, linalg.LinAlgError) as e:
            diag = 1 / (grads**2).mean(0)
            diag = np.clip(diag, 1e-6, 1e6)
            self._tmp_cov = np.diag(diag)
            self._tmp_chol = np.diag(np.sqrt(diag))
            if self.verbose:
                print(e)
                print("diag:", np.diag(self._tmp_cov))
            self._chol_error = e

    def stats(self):
        stats = super().stats().copy()
        if self._cov is not None:
            vals = self._cov.eigvals_
            if len(vals) > 0:
                stats.update(
                    {"largest_eigval": max(vals), "smallest_eigval": min(vals)}
                )
        return stats

    def update(self, sample, grad, tune):
        if not tune:
            return

        # Steps since previous update
        delta = self._n_samples - self._previous_update

        if self._n_samples > self._adapt_stop:
            pass
        elif self._n_samples == self._adapt_stop:
            self._foreground[0].append(sample.copy())
            self._foreground[1].append(grad.copy())
            self._background[0].append(sample.copy())
            self._background[1].append(grad.copy())

            if self.verbose:
                print(
                    self._n_samples,
                    f"Final update with {len(self._foreground[0])} draws",
                )
            self._update_from_weightvar(self._foreground, soft=False)
        elif self._n_samples > self._discard_window:
            self._foreground[0].append(sample.copy())
            self._foreground[1].append(grad.copy())
            self._background[0].append(sample.copy())
            self._background[1].append(grad.copy())

            # Reset the background covariance if we are at the end of the adaptation
            # window.
            if delta >= self._adaptation_window:
                # print(self._n_samples, "next window")
                if self.verbose:
                    print(
                        self._n_samples,
                        f"Window switch update with {len(self._background[0])} draws",
                    )
                self._foreground = self._background
                self._background = ([], [])

                self._previous_update = self._n_samples
                self._adaptation_window = int(
                    self._adaptation_window * self._adaptation_window_multiplier
                )
                self._update_from_weightvar(self._foreground, soft=False)

            # Update the covariance matrix and recompute the Cholesky factorization
            # every "update_window" steps
            elif (delta + 1) % self._update_window == 0:
                if self.verbose:
                    print(
                        self._n_samples,
                        f"Small cov update with {len(self._foreground[0])} draws",
                    )
                self._update_from_weightvar(self._foreground, soft=True)

        self._n_samples += 1

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError(str(self._chol_error))

    def velocity(self, x, out=None):
        """Compute the current velocity at a position in parameter space."""
        if self._cov is None:
            return np.dot(self._initial_cov, x, out=out)

        result = self._cov.matmul_single(x)
        if out is not None:
            out[...] = result
        return result

    def random(self):
        """Draw random value from QuadPotential."""
        if self._cov is None:
            vals = np.random.normal(size=self._n).astype(self.dtype)
            return scipy.linalg.solve_triangular(
                self._initial_chol.T, vals, overwrite_b=True
            )

        return self._cov.draw_inv()
