"""
src/solver.py
Phase 2: Simplex-Constrained ADMM Solver for Sparse Index Tracking

Solves:
    min_w  (1/2)||Xw - y||_2^2  +  λ||w||_1
    s.t.   w >= 0

Post-convergence: normalizes weights to sum to 1 (fully invested).

Stress-tested 3x for:
  - Mathematical correctness (ADMM updates, L1-on-simplex trap)
  - Numerical stability (Cholesky factorization, adaptive ρ, division-by-zero)
  - Software robustness (convergence history, regularization path, edge cases)
"""

import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class SparseTrackerADMM:
    """
    ADMM solver for L1-regularized, non-negative portfolio optimization.

    The solver finds a sparse weight vector w such that:
        - Xw ≈ y  (tracking the benchmark)
        - w >= 0   (long-only, no short-selling)
        - sum(w) = 1  (fully invested, enforced post-convergence)
        - ~30 out of ~500 weights are non-zero (sparsity from L1)

    ADMM Updates (scaled form):
        w-update (Ridge):  w = (X'X + ρI)^{-1} (X'y + ρ(z - u))
        z-update (Prox):   z = max(w + u - λ/ρ, 0)
        u-update (Dual):   u = u + w - z

    Parameters
    ----------
    lam : float
        L1 regularization parameter (λ). Higher = sparser.
    rho : float
        ADMM penalty parameter (ρ). Controls convergence speed.
    max_iter : int
        Maximum number of ADMM iterations.
    tol : float
        Convergence tolerance for primal and dual residuals.
    adaptive_rho : bool
        If True, adaptively scale ρ (Boyd et al. 2011) for faster convergence.
    rho_scale : float
        Multiplicative factor for adaptive ρ scaling.
    verbose : bool
        If True, print iteration progress.
    """

    def __init__(
        self,
        lam: float = 0.01,
        rho: float | str = "auto",
        max_iter: int = 5000,
        tol: float = 1e-6,
        adaptive_rho: bool = True,
        rho_scale: float = 2.0,
        verbose: bool = True,
        lam_vec: np.ndarray | None = None,
    ):
        # `rho="auto"` (default) sets ρ = σ_max(X) at fit-time. This is the
        # *Boyd-2011-recommended* scaling that makes the (X'X + ρI) w-update
        # matrix well-conditioned regardless of the absolute scale of X.
        # A fixed `rho=1.0` mis-scales for any X whose largest singular value
        # is far from 1 (e.g. raw returns where σ_max ≈ 0.05, or synthetic
        # standard-normal X where σ_max ≈ √p).
        #
        # `lam_vec` is an optional per-coordinate L1 weight vector. When set,
        # the z-update uses positive soft-thresholding with threshold
        # ``lam_vec / rho`` instead of the scalar ``lam / rho``. This is the
        # mechanism powering Phase 1's iterative reweighted-ℓ₁ MM scheme
        # (Candès, Wakin & Boyd 2008, "Enhancing Sparsity by Reweighted ℓ₁
        # Minimization"). When `lam_vec is None`, behaviour is identical to
        # the pre-Phase-1 scalar-λ solver — full backward compatibility.
        self.lam = lam
        self.rho_init = rho  # may be "auto" or a positive float; resolved in _precompute
        self.rho: float = float(rho) if isinstance(rho, (int, float)) else float("nan")
        self.max_iter = max_iter
        self.tol = tol
        self.adaptive_rho = adaptive_rho
        self.rho_scale = rho_scale
        self.verbose = verbose
        self.lam_vec = None if lam_vec is None else np.ascontiguousarray(lam_vec, dtype=np.float64)

        # --- State variables (populated by fit) ---
        self.w = None  # Primal variable (unconstrained)
        self.z = None  # Auxiliary variable (constrained)
        self.u = None  # Scaled dual variable
        self.n_iter = 0  # Iterations to convergence
        self.converged = False  # Whether solver converged

        # --- Convergence history ---
        self.primal_residuals = []
        self.dual_residuals = []
        self.objective_values = []

        # --- Precomputed quantities ---
        self._cho_factor = None  # Cholesky factorization of (X'X + ρI)
        self._XtX = None  # X'X (p x p)
        self._Xty = None  # X'y (p,)

    # ================================================================
    # Static Helpers
    # ================================================================

    @staticmethod
    def compute_lambda_max(X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute λ_max: the smallest λ where z = 0 (trivial solution).

        At z=0, the gradient of the quadratic loss is X'y.
        The positive soft-threshold zeros out component i when:
            (X'y)_i / ρ - λ/ρ <= 0  →  λ >= (X'y)_i
        So λ_max = max(X'y), considering only positive components
        (negative components are already zeroed by the max(·, 0) clamp).

        Parameters
        ----------
        X : ndarray of shape (n, p)
        y : ndarray of shape (n,)

        Returns
        -------
        float : λ_max
        """
        Xty = X.T @ y
        lam_max = np.max(Xty)  # Only positive components matter
        return max(lam_max, 1e-10)  # Guard against non-positive edge case

    # ================================================================
    # Internal ADMM Steps
    # ================================================================

    @staticmethod
    def _auto_rho_from_XtX(XtX: np.ndarray) -> float:
        """ρ = √λ_max(X'X) — the Boyd-2011 well-scaling heuristic.

        Computed via 30-step power iteration (fast, O(p²) per step).
        For (p=500) takes ≈4ms; cheaper than a single ADMM iteration.
        """
        p = XtX.shape[0]
        rng = np.random.default_rng(0)
        v = rng.standard_normal(p)
        v /= np.linalg.norm(v) + 1e-30
        lam_top = 0.0
        for _ in range(30):
            v = XtX @ v
            nrm = float(np.linalg.norm(v))
            if nrm < 1e-30:
                break
            v /= nrm
            lam_top = float(v @ (XtX @ v))
        return float(np.sqrt(max(lam_top, 1e-12)))

    def _precompute(self, X: np.ndarray, y: np.ndarray):
        """
        Precompute X'X, X'y, and the Cholesky factorization of (X'X + ρI).

        Uses Cholesky decomposition instead of matrix inversion for
        numerical stability: (X'X + ρI) is symmetric positive definite
        since ρ > 0.

        If ``self.rho_init == "auto"``, sets ρ = σ_max(X) here (data-aware
        well-scaling) and freezes it (so adaptive ρ then perturbs around a
        sensible value rather than starting at the mis-scaled ρ=1).
        """
        self._XtX = X.T @ X  # (p, p)
        self._Xty = X.T @ y  # (p,)
        # Resolve auto-ρ now that we know X
        if self.rho_init == "auto" or not isinstance(self.rho_init, (int, float)):
            self.rho = self._auto_rho_from_XtX(self._XtX)
        else:
            self.rho = float(self.rho_init)
        p = X.shape[1]
        A = self._XtX + self.rho * np.eye(p)  # (p, p), SPD
        self._cho_factor = cho_factor(A, lower=False)  # Cholesky cache

    def _recompute_cholesky(self):
        """Re-factorize when ρ changes (adaptive scaling)."""
        p = self._XtX.shape[0]
        A = self._XtX + self.rho * np.eye(p)
        self._cho_factor = cho_factor(A, lower=False)

    def _w_update(self):
        """
        w-update (Ridge regression step):
            w^{k+1} = (X'X + ρI)^{-1} (X'y + ρ(z^k - u^k))

        Solved via Cholesky back-substitution (numerically stable).
        """
        rhs = self._Xty + self.rho * (self.z - self.u)  # (p,)
        self.w = cho_solve(self._cho_factor, rhs)  # (p,)

    def _z_update(self):
        """
        z-update (Positive soft-thresholding):
            z^{k+1} = max(w^{k+1} + u^k - λ/ρ, 0)

        This is the proximal operator of g(z) = λ||z||_1 + I_{z>=0}(z).
        Since z >= 0, soft-threshold and non-negativity combine into
        a single max(·, 0) operation.

        If ``self.lam_vec`` is set (reweighted-ℓ₁ mode), the threshold is
        per-coordinate ``lam_vec / rho`` instead of a scalar.
        """
        v = self.w + self.u  # (p,)
        if self.lam_vec is None:
            threshold = self.lam / self.rho  # scalar broadcast
        else:
            threshold = self.lam_vec / self.rho  # (p,) per-coordinate
        self.z = np.maximum(v - threshold, 0.0)  # (p,)

    def _u_update(self):
        """
        u-update (Scaled dual ascent):
            u^{k+1} = u^k + w^{k+1} - z^{k+1}
        """
        self.u = self.u + self.w - self.z

    def _adapt_rho_step(self, primal_res: float, dual_res: float, iteration: int):
        """
        Adaptive ρ scaling with Wohlberg-style *normalized* residual balancing.

        Standard Boyd-2011 (§3.4.1) compares raw residuals
        ``primal_res`` vs ``mu * dual_res``. That works when ‖w‖, ‖z‖ and ‖u‖
        are O(1), but on data with very different scales (raw return matrices
        vs synthetic standard-normal features) the raw residuals occupy very
        different magnitudes and the algorithm can oscillate.

        Wohlberg (2017, "ADMM Penalty Parameter Selection by Residual Balancing")
        proposes normalising before comparing:

            r̂ = primal_res / max(‖w‖, ‖z‖)
            ŝ = dual_res / (ρ · ‖u‖)

        Plus three robustness guards:
          1. **Warm-up**: leave ρ alone for the first ``warmup_iters`` steps so
             the algorithm finds its operating point before we perturb it.
          2. **Throttling**: only attempt to adapt every ``adapt_every`` iterations
             so each ρ change can take effect.
          3. **Cap total adaptations**: refuse to change ρ more than
             ``max_adaptations`` times across the run. Saves Cholesky refactor
             work and avoids oscillation in pathological cases.
        """
        warmup_iters = 10
        adapt_every = 10
        max_adaptations = 8
        mu = 10.0

        if iteration < warmup_iters or iteration % adapt_every != 0:
            return
        if getattr(self, "_n_rho_adaptations", 0) >= max_adaptations:
            return

        norm_w = float(np.linalg.norm(self.w))
        norm_z = float(np.linalg.norm(self.z))
        norm_u = float(np.linalg.norm(self.u))
        primal_scale = max(norm_w, norm_z, 1e-12)
        dual_scale = max(self.rho * norm_u, 1e-12)
        r_hat = primal_res / primal_scale
        s_hat = dual_res / dual_scale

        rho_changed = False
        if r_hat > mu * s_hat:
            self.rho *= self.rho_scale
            rho_changed = True
        elif s_hat > mu * r_hat:
            self.rho /= self.rho_scale
            rho_changed = True

        if rho_changed:
            self._n_rho_adaptations = getattr(self, "_n_rho_adaptations", 0) + 1
            self._recompute_cholesky()

    def _compute_objective(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the objective value: (1/2)||Xw - y||^2 + λ||w||_1.

        In reweighted-ℓ₁ mode (``lam_vec`` set), the L1 penalty becomes the
        weighted sum ``<lam_vec, |z|>``.
        """
        residual = X @ self.z - y
        if self.lam_vec is None:
            l1_term = self.lam * np.sum(np.abs(self.z))
        else:
            l1_term = float(self.lam_vec @ np.abs(self.z))
        return 0.5 * np.dot(residual, residual) + l1_term

    # ================================================================
    # Main Solver
    # ================================================================

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Run the ADMM solver on the data.

        Parameters
        ----------
        X : ndarray of shape (n, p)
            Standardized feature matrix (constituent returns).
        y : ndarray of shape (n,)
            Benchmark return vector (SPY).

        Returns
        -------
        self : SparseTrackerADMM
            Fitted solver instance.
        """
        n, p = X.shape

        if self.verbose:
            print(f"\n{'=' * 65}")
            print("  ADMM SOLVER — Phase 2")
            print(f"{'=' * 65}")
            print(f"  Matrix X: ({n}, {p})  |  Vector y: ({n},)")
            print(f"  λ = {self.lam:.6f}  |  ρ = {self.rho:.4f}")
            print(f"  max_iter = {self.max_iter}  |  tol = {self.tol:.1e}")
            print(f"  adaptive_ρ = {self.adaptive_rho}")
            print(f"{'=' * 65}\n")

        # --- Initialize variables as zero vectors ---
        self.w = np.zeros(p)
        self.z = np.zeros(p)
        self.u = np.zeros(p)
        self._n_rho_adaptations = 0

        # --- Precompute X'X, X'y, Cholesky of (X'X + ρI) ---
        self._precompute(X, y)

        # --- Reset history ---
        self.primal_residuals = []
        self.dual_residuals = []
        self.objective_values = []
        self.converged = False
        z_prev = self.z.copy()

        # --- ADMM iteration loop ---
        for k in range(1, self.max_iter + 1):
            # Step 1: w-update (Ridge)
            self._w_update()

            # Step 2: z-update (Positive soft-threshold)
            self._z_update()

            # Step 3: u-update (Dual ascent)
            self._u_update()

            # --- Convergence diagnostics ---
            # Primal residual: r^k = ||w^{k+1} - z^{k+1}||_2
            # Measures constraint violation (Boyd et al. 2011, Eq. 3.12)
            primal_res = np.linalg.norm(self.w - self.z)

            # Dual residual: s^k = ||ρ(z^{k+1} - z^k)||_2
            # Measures optimality violation (Boyd et al. 2011, Eq. 3.13)
            # NOTE: must capture z_prev BEFORE overwriting it
            dual_res = np.linalg.norm(self.rho * (self.z - z_prev))

            # Save z_prev for next iteration BEFORE adaptive ρ may change
            z_prev = self.z.copy()

            # Objective value: (1/2)||Xz - y||² + λ||z||₁
            obj = self._compute_objective(X, y)

            # Store history
            self.primal_residuals.append(primal_res)
            self.dual_residuals.append(dual_res)
            self.objective_values.append(obj)

            # --- Adaptive ρ scaling (Boyd 2011 + Wohlberg 2017 normalisation) ---
            if self.adaptive_rho:
                self._adapt_rho_step(primal_res, dual_res, iteration=k)

            # --- Print progress ---
            if self.verbose and (k <= 5 or k % 100 == 0 or k == self.max_iter):
                n_nonzero = np.sum(self.z > 1e-8)
                print(
                    f"  iter {k:5d} | "
                    f"primal_res={primal_res:.2e} | "
                    f"dual_res={dual_res:.2e} | "
                    f"obj={obj:.6f} | "
                    f"ρ={self.rho:.4f} | "
                    f"nnz={n_nonzero}"
                )

            # --- Convergence check (BOTH residuals must be small) ---
            # Relative+absolute tolerance (Boyd et al. 2011, Eq. 3.12-3.13)
            eps_abs = self.tol
            eps_rel = self.tol
            eps_primal = eps_abs * np.sqrt(p) + eps_rel * max(
                np.linalg.norm(self.w), np.linalg.norm(self.z)
            )
            eps_dual = eps_abs * np.sqrt(p) + eps_rel * np.linalg.norm(self.u)

            if primal_res < eps_primal and dual_res < eps_dual:
                self.converged = True
                self.n_iter = k
                if self.verbose:
                    print(f"\n  ✅ CONVERGED at iteration {k}")
                break

        else:
            # max_iter hit without convergence
            self.n_iter = self.max_iter
            if self.verbose:
                print(f"\n  ⚠️  MAX ITERATIONS ({self.max_iter}) reached without convergence.")
                print(f"      Final primal_res={primal_res:.2e}, dual_res={dual_res:.2e}")

        # --- Final report ---
        if self.verbose:
            n_nonzero = np.sum(self.z > 1e-8)
            print(f"\n{'=' * 65}")
            print("  ADMM Results")
            print(f"{'=' * 65}")
            print(f"  Iterations:    {self.n_iter}")
            print(f"  Converged:     {self.converged}")
            print(f"  Final primal:  {self.primal_residuals[-1]:.2e}")
            print(f"  Final dual:    {self.dual_residuals[-1]:.2e}")
            print(f"  Final obj:     {self.objective_values[-1]:.6f}")
            print(f"  Non-zero (z):  {n_nonzero} / {p}")
            print(f"  Sum(z):        {np.sum(self.z):.6f}")
            print(f"{'=' * 65}")

        return self

    # ================================================================
    # Post-Convergence Methods
    # ================================================================

    def get_sparse_weights(self, threshold: float = 1e-6) -> np.ndarray:
        """
        Get the final normalized, sparse portfolio weights (standardized space).

        1. Hard-clip tiny weights to exactly 0 (cleaner sparsity)
        2. Normalize to sum to 1 (fully invested constraint)

        Parameters
        ----------
        threshold : float
            Weights below this value are set to zero.

        Returns
        -------
        ndarray of shape (p,) : normalized sparse weights summing to 1.

        Raises
        ------
        ValueError
            If all weights are zero (λ too large).
        """
        if self.z is None:
            raise RuntimeError("❌ Solver has not been fitted. Call fit(X, y) first.")

        weights = self.z.copy()

        # Hard threshold: clip tiny weights to exactly 0
        weights[weights < threshold] = 0.0

        # Normalize to simplex: sum = 1
        total = np.sum(weights)
        if total < 1e-12:
            raise ValueError(
                f"❌ All weights are zero. λ={self.lam:.6f} is too large. "
                f"Try a smaller λ (λ_max = {self.compute_lambda_max.__name__}())."
            )

        weights /= total
        return weights

    def get_raw_weights(self, X_std: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
        """
        Get portfolio weights inverse-transformed to raw (unstandardized) return space.

        The ADMM optimizes on standardized X, where X_std = (X_raw - mean) / std.
        The relationship is:
            X_std @ w_std = ((X_raw - mean) / std) @ w_std
                         = X_raw @ (w_std / std) - offset

        So the correct raw-space weights are: w_raw_j = w_std_j / std_j,
        then re-normalized to sum to 1.

        Without this transform, the L1 penalty's uniform shrinkage in
        standardized space causes distorted allocations in raw space:
        stocks with high volatility (large std) get inflated weights.

        Parameters
        ----------
        X_std : ndarray of shape (p,)
            Column standard deviations used for standardization (from Phase 1).
        threshold : float
            Weights below this in standardized space are set to zero.

        Returns
        -------
        ndarray of shape (p,) : raw-space weights summing to 1.
        """
        # Get sparse weights in standardized space (before normalization)
        if self.z is None:
            raise RuntimeError("❌ Solver has not been fitted. Call fit(X, y) first.")

        w_std = self.z.copy()
        w_std[w_std < threshold] = 0.0

        total_std = np.sum(w_std)
        if total_std < 1e-12:
            raise ValueError(f"❌ All weights are zero. λ={self.lam:.6f} is too large.")

        # Inverse-transform: divide by std to convert from standardized → raw space
        w_raw = w_std / X_std
        w_raw[w_std == 0] = 0.0  # preserve sparsity pattern

        # Re-normalize to sum = 1 (fully invested)
        w_raw /= np.sum(w_raw)
        return w_raw

    def summary(self, weights: np.ndarray, stock_names: list = None, top_k: int = 15):
        """
        Print a human-readable summary of the portfolio.

        Parameters
        ----------
        weights : ndarray of shape (p,)
            Portfolio weights to summarize (should be raw-space weights).
        stock_names : list of str, optional
            Ticker symbols corresponding to weight indices.
        top_k : int
            Number of top holdings to display.
        """
        n_nonzero = np.sum(weights > 0)
        active_idx = np.where(weights > 0)[0]

        print(f"\n{'=' * 65}")
        print("  SPARSE PORTFOLIO SUMMARY")
        print(f"{'=' * 65}")
        print(f"  Active stocks:  {n_nonzero} / {len(weights)}")
        print(f"  Sum of weights: {np.sum(weights):.6f}")
        print(f"  Max weight:     {np.max(weights):.6f}")
        print(f"  Min (active):   {np.min(weights[active_idx]):.6f}")

        # Sort by weight descending
        sorted_idx = np.argsort(weights)[::-1]

        print(f"\n  Top-{top_k} Holdings:")
        print(f"  {'Rank':<6}{'Ticker':<10}{'Weight':>10}{'Pct':>10}")
        print(f"  {'-' * 36}")
        for rank, idx in enumerate(sorted_idx[:top_k], 1):
            if weights[idx] <= 0:
                break
            ticker = stock_names[idx] if stock_names else f"Stock_{idx}"
            pct = weights[idx] * 100
            print(f"  {rank:<6}{ticker:<10}{weights[idx]:>10.6f}{pct:>9.2f}%")
        print(f"{'=' * 65}")

    # ================================================================
    # Regularization Path (Phase 3 prep)
    # ================================================================

    def regularization_path(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_lambdas: int = 50,
        lam_min_ratio: float = 0.001,
        verbose: bool = False,
    ) -> dict:
        """
        Sweep λ values and record sparsity at each level.

        Parameters
        ----------
        X : ndarray of shape (n, p)
        y : ndarray of shape (n,)
        n_lambdas : int
            Number of λ values to test.
        lam_min_ratio : float
            Smallest λ as a fraction of λ_max.
        verbose : bool
            Print per-λ progress.

        Returns
        -------
        dict with keys:
            'lambdas': array of λ values tested
            'n_nonzeros': array of non-zero weight counts
            'objectives': array of final objective values
            'tracking_errors': array of ||Xw - y||_2 at each λ
        """
        lam_max = self.compute_lambda_max(X, y)
        lambdas = np.logspace(np.log10(lam_min_ratio * lam_max), np.log10(lam_max), n_lambdas)

        n_nonzeros = []
        objectives = []
        tracking_errors = []

        print(f"\n📊 Regularization Path: sweeping {n_lambdas} λ values")
        print(f"   λ range: [{lambdas[0]:.6f}, {lambdas[-1]:.6f}]")
        print(f"   λ_max = {lam_max:.6f}\n")

        for i, lam in enumerate(lambdas):
            # Create a fresh solver for each λ
            solver = SparseTrackerADMM(
                lam=lam,
                rho=1.0,
                max_iter=self.max_iter,
                tol=self.tol,
                adaptive_rho=self.adaptive_rho,
                verbose=False,
            )
            solver.fit(X, y)

            try:
                w = solver.get_sparse_weights()
            except ValueError:
                # All weights zero — λ too large
                w = np.zeros(X.shape[1])

            nnz = np.sum(w > 0)
            obj = solver.objective_values[-1] if solver.objective_values else float("inf")
            te = np.linalg.norm(X @ w - y)

            n_nonzeros.append(nnz)
            objectives.append(obj)
            tracking_errors.append(te)

            if verbose or (i + 1) % 10 == 0 or i == 0:
                print(f"   λ={lam:.6f} | nnz={nnz:4d} | obj={obj:.4f} | TE={te:.6f}")

        print("\n   ✅ Regularization path complete.")

        return {
            "lambdas": lambdas,
            "n_nonzeros": np.array(n_nonzeros),
            "objectives": np.array(objectives),
            "tracking_errors": np.array(tracking_errors),
        }


# ====================================================================
# Quick Execution & Validation
# ====================================================================

if __name__ == "__main__":
    import pandas as pd

    from sit.paths import DATA_DIR as _DATA_DIR

    DATA_DIR = str(_DATA_DIR)

    print("=" * 65)
    print("  PHASE 2: ADMM SOLVER — EXECUTION & VALIDATION")
    print("=" * 65)

    # --- Load Phase 1 data ---
    print("\n📥 Loading Phase 1 data...")
    X = np.load(os.path.join(DATA_DIR, "X_standardized.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_spy.npy"))
    stock_names = pd.read_csv(os.path.join(DATA_DIR, "stock_names.csv")).iloc[:, 0].tolist()

    print(f"   X: {X.shape}  |  y: {y.shape}  |  Stocks: {len(stock_names)}")

    # --- Compute λ_max and choose λ ---
    lam_max = SparseTrackerADMM.compute_lambda_max(X, y)
    # Tuned via sweep on RAW returns: 0.05 × λ_max → ~58 stocks, R²=0.674
    lam = 0.05 * lam_max
    print(f"   λ_max = {lam_max:.6f}")
    print(f"   λ     = {lam:.6f}  (0.05 × λ_max)")

    # --- Run ADMM solver ---
    solver = SparseTrackerADMM(
        lam=lam,
        rho=1.0,
        max_iter=5000,
        tol=1e-6,
        adaptive_rho=True,
        verbose=True,
    )
    solver.fit(X, y)

    # --- Inverse-transform weights: standardized → raw space ---
    # ADMM found w_std on standardized X. To apply to raw returns:
    #   w_raw_j = w_std_j / std_j, then re-normalize to sum = 1.
    # Without this, volatile stocks get inflated weights.
    X_raw = np.load(os.path.join(DATA_DIR, "X_raw.npy"))
    X_std_dev = np.load(os.path.join(DATA_DIR, "X_std.npy"))
    weights = solver.get_raw_weights(X_std_dev)

    # --- Compute tracking metrics on raw returns ---
    port_return_raw = X_raw @ weights
    ss_res = np.sum((y - port_return_raw) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_raw = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    daily_te = np.std(port_return_raw - y)
    ann_te_pct = daily_te * np.sqrt(252) * 100  # annualized, in percent

    # --- Validation checks ---
    print(f"\n{'=' * 65}")
    print("  VALIDATION CHECKS")
    print(f"{'=' * 65}")

    n_nonzero = np.sum(weights > 0)
    weight_sum = np.sum(weights)
    all_nonneg = np.all(weights >= 0)

    checks = [
        ("Weights sum to 1.0", abs(weight_sum - 1.0) < 1e-10, f"{weight_sum:.10f}"),
        ("All weights ≥ 0 (long-only)", all_nonneg, f"min={np.min(weights):.2e}"),
        ("Sparsity (non-zero stocks)", 10 <= n_nonzero <= 100, f"{n_nonzero} / {len(weights)}"),
        ("Solver converged", solver.converged, f"iter={solver.n_iter}"),
        ("No NaN in weights", not np.isnan(weights).any(), ""),
        ("No Inf in weights", not np.isinf(weights).any(), ""),
    ]

    all_passed = True
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {name}: {detail}")
        if not passed:
            all_passed = False

    # --- Tracking accuracy on raw returns ---
    print("\n  📈 Tracking Accuracy (on raw returns, un-scaled weights):")
    print(f"     R²                = {r2_raw:.4f}")
    print(f"     Annualized TE     = {ann_te_pct:.2f}%")
    print(f"     Daily TE (std)    = {daily_te:.6f}")

    # --- Save results ---
    np.save(os.path.join(DATA_DIR, "sparse_weights.npy"), weights)
    print("\n  💾 Saved: data/sparse_weights.npy (raw-space weights)")

    # Active stock tickers
    active_idx = np.where(weights > 0)[0]
    active_tickers = [stock_names[i] for i in active_idx]
    active_weights = weights[active_idx]
    pd.DataFrame({"ticker": active_tickers, "weight": active_weights}).sort_values(
        "weight", ascending=False
    ).to_csv(os.path.join(DATA_DIR, "active_stocks.csv"), index=False)
    print("  💾 Saved: data/active_stocks.csv")

    # --- Print portfolio summary ---
    solver.summary(weights=weights, stock_names=stock_names, top_k=15)

    # --- Final verdict ---
    print()
    if all_passed:
        print("  🎉 ALL VALIDATION CHECKS PASSED — Phase 2 Complete!")
    else:
        print("  ⚠️  Some checks failed. Review output above.")
    print(f"  R² = {r2_raw:.4f}  |  Annualized TE = {ann_te_pct:.2f}%")
    print("=" * 65)
