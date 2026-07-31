# =============================================================================
# THREAD CONFIGURATION — Must appear before any numpy/scipy/sklearn imports
# to prevent CPU core over-subscription during joblib parallel LOOCV execution.
# =============================================================================
import os
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"

# =============================================================================
# Standard imports (after env overrides)
# =============================================================================
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# =============================================================================
# Module-level worker — defined at top level so joblib 'loky' backend can
# pickle it without closure overhead.
#
# Mathematical state:
#   L_x1 : (N_total, k_x - 1)  — unified LDA projection in X-space
#   L_y1 : (N_total, k_y - 1)  — unified LDA projection in Y-space
#   v    : int in [0, N_total)  — held-out sample index for this LOOCV fold
# =============================================================================
def _validate_single_sample(
    v: int,
    L_x1: np.ndarray,
    L_y1: np.ndarray,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Single LOOCV fold worker.

    Parameters
    ----------
    v     : Held-out sample index.
    L_x1  : Unified LDA latent matrix in X-space, shape (N_total, k_x - 1).
    L_y1  : Unified LDA latent matrix in Y-space, shape (N_total, k_y - 1).

    Returns
    -------
    (v, Z_x_v, Z_y_v) where Z_x_v and Z_y_v are the out-of-sample CCA
    projections for fold v, each of shape (d_cca,).
    """
    # --- Build leave-one-out training masks -----------------------------------
    mask = np.ones(L_x1.shape[0], dtype=bool)
    mask[v] = False

    L_x_train = L_x1[mask]   # (N_total - 1, k_x - 1)
    L_y_train = L_y1[mask]   # (N_total - 1, k_y - 1)

    # --- Fit CCA on the training split ---------------------------------------
    # d_cca = min(k_x - 1, k_y - 1) canonical dimensions are learned.
    d_cca = min(L_x_train.shape[1], L_y_train.shape[1])
    cca = CCA(n_components=d_cca, max_iter=1000)
    cca.fit(L_x_train, L_y_train)

    # Canonical weight matrices: V_x_val (k_x-1, d_cca), V_y_val (k_y-1, d_cca)
    V_x_val = cca.x_rotations_   # shape: (k_x - 1, d_cca)
    V_y_val = cca.y_rotations_   # shape: (k_y - 1, d_cca)

    # --- Project the single held-out row via explicit matrix multiply ---------
    # L_x1[v] : (k_x - 1,)  =>  Z_x_v : (d_cca,)
    Z_x_v = L_x1[v] @ V_x_val   # (d_cca,)
    Z_y_v = L_y1[v] @ V_y_val   # (d_cca,)

    return v, Z_x_v, Z_y_v


# =============================================================================
# MorphologicalCoregistrationEngine
# =============================================================================
class MorphologicalCoregistrationEngine:
    """
    Model-1 — Morphological Coregistration Engine.

    Performs paired-feature, leave-one-out cross-validated latent space
    alignment between two datasets that share an identical feature dimension
    (m_shared) but may have completely independent sample sizes (n_x, n_y).

    Pipeline
    --------
    A. GMM clustering  →  hard label vectors C_x_0, C_y_0
    B. LDA projection  →  unified latent spaces L_x1, L_y1  (N_total rows each)
    C. Parallel LOOCV CCA  →  cross-validated matrices Z_x, Z_y  (N_total, d_cca)
    D. Alignment error diagnostics  +  .npy export of Z_morph

    Parameters
    ----------
    k_x      : Number of GMM/LDA components for Dataset X.
    k_y      : Number of GMM/LDA components for Dataset Y.
    gmm_seed : Random seed for reproducible GMM initialisation.
    n_jobs   : Passed directly to joblib.Parallel (-1 = all logical cores).
    output_dir : Directory to write the inter-repository .npy artefact.
    """

    def __init__(
        self,
        k_x: int = 5,
        k_y: int = 5,
        gmm_seed: int = 42,
        n_jobs: int = -1,
        output_dir: str = "outputs",
    ) -> None:
        self.k_x        = k_x
        self.k_y        = k_y
        self.gmm_seed   = gmm_seed
        self.n_jobs     = n_jobs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Artefacts populated by fit()
        self.Z_morph:  np.ndarray | None = None  # (N_total, d_cca)
        self.Z_x:      np.ndarray | None = None
        self.Z_y:      np.ndarray | None = None
        self.L_x1:     np.ndarray | None = None
        self.L_y1:     np.ndarray | None = None
        self.E_train:  float | None = None
        self.E_val:    float | None = None

    # -------------------------------------------------------------------------
    # Step A — GMM Clustering
    # -------------------------------------------------------------------------
    def _step_a_gmm_clustering(
        self,
        X_shared: np.ndarray,  # (n_x, m_shared)
        Y_shared: np.ndarray,  # (n_y, m_shared)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit independent GMMs on X_shared and Y_shared.

        Returns
        -------
        C_x_0 : Hard cluster labels for X, shape (n_x,)
        C_y_0 : Hard cluster labels for Y, shape (n_y,)
        """
        log.info("Step A — GMM clustering: k_x=%d, k_y=%d", self.k_x, self.k_y)

        gmm_x = GaussianMixture(
            n_components=self.k_x,
            covariance_type="full",
            random_state=self.gmm_seed,
        )
        gmm_y = GaussianMixture(
            n_components=self.k_y,
            covariance_type="full",
            random_state=self.gmm_seed,
        )

        C_x_0 = gmm_x.fit_predict(X_shared)  # (n_x,)
        C_y_0 = gmm_y.fit_predict(Y_shared)  # (n_y,)

        log.info("  C_x_0 unique labels: %s", np.unique(C_x_0))
        log.info("  C_y_0 unique labels: %s", np.unique(C_y_0))
        return C_x_0, C_y_0

    # -------------------------------------------------------------------------
    # Step B — 0th-Epoch LDA Projection → Unified Spaces L_x1, L_y1
    # -------------------------------------------------------------------------
    def _step_b_lda_projection(
        self,
        X_shared: np.ndarray,  # (n_x, m_shared)
        Y_shared: np.ndarray,  # (n_y, m_shared)
        C_x_0: np.ndarray,     # (n_x,)
        C_y_0: np.ndarray,     # (n_y,)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit LDA on each dataset, extract weight matrices W and biases b,
        then build unified cross-projected latent spaces.

        Mathematical state
        ------------------
        W_x1 : (m_shared, k_x - 1)  — LDA scalings from X-fit
        b_x1 : (k_x - 1,)           — LDA intercept from X-fit
        W_y1 : (m_shared, k_y - 1)  — LDA scalings from Y-fit
        b_y1 : (k_y - 1,)           — LDA intercept from Y-fit

        L_x1 = vstack([ X @ W_x1 + b_x1,  Y @ W_x1 + b_x1 ])  # (N_total, k_x-1)
        L_y1 = vstack([ X @ W_y1 + b_y1,  Y @ W_y1 + b_y1 ])  # (N_total, k_y-1)

        Returns
        -------
        L_x1 : (N_total, k_x - 1)
        L_y1 : (N_total, k_y - 1)
        """
        log.info("Step B — LDA projection (0th epoch).")

        lda_x = LinearDiscriminantAnalysis()
        lda_y = LinearDiscriminantAnalysis()

        lda_x.fit(X_shared, C_x_0)
        lda_y.fit(Y_shared, C_y_0)

        # Extract weight matrices via scalings_ (m_shared, n_components)
        # sklearn stores scalings_ as (m_shared, min(k-1, m_shared)).
        W_x1 = lda_x.scalings_   # (m_shared, k_x - 1)
        W_y1 = lda_y.scalings_   # (m_shared, k_y - 1)

        # Intercepts: xbar projected through W gives the class-mean offset.
        # We use the grand mean of each LDA's class means as the bias.
        b_x1 = lda_x.xbar_ @ W_x1   # (k_x - 1,)
        b_y1 = lda_y.xbar_ @ W_y1   # (k_y - 1,)

        # --- Within-space + cross-space projections via @ operator ----------
        # L_x1 : project both X and Y through X-LDA weights
        L_x1 = np.vstack([
            (X_shared @ W_x1) + b_x1,   # (n_x, k_x - 1)
            (Y_shared @ W_x1) + b_x1,   # (n_y, k_x - 1)
        ])                               # → (N_total, k_x - 1)

        # L_y1 : project both X and Y through Y-LDA weights
        L_y1 = np.vstack([
            (X_shared @ W_y1) + b_y1,   # (n_x, k_y - 1)
            (Y_shared @ W_y1) + b_y1,   # (n_y, k_y - 1)
        ])                               # → (N_total, k_y - 1)

        log.info("  L_x1 shape: %s", L_x1.shape)
        log.info("  L_y1 shape: %s", L_y1.shape)
        return L_x1, L_y1

    # -------------------------------------------------------------------------
    # Step C — Parallel CPU LOOCV CCA Engine
    # -------------------------------------------------------------------------
    def _step_c_loocv_cca(
        self,
        L_x1: np.ndarray,  # (N_total, k_x - 1)
        L_y1: np.ndarray,  # (N_total, k_y - 1)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run N_total LOOCV folds in parallel via joblib (loky backend).

        Each fold v:
          - Trains CCA on all rows except v  →  V_x_val, V_y_val
          - Projects row v  →  Z_x_v = L_x1[v] @ V_x_val
                               Z_y_v = L_y1[v] @ V_y_val

        Returns
        -------
        Z_x : (N_total, d_cca)  — out-of-sample canonical scores in X-space
        Z_y : (N_total, d_cca)  — out-of-sample canonical scores in Y-space
        """
        N_total = L_x1.shape[0]
        d_cca   = min(L_x1.shape[1], L_y1.shape[1])
        log.info(
            "Step C — Parallel LOOCV CCA: N_total=%d, d_cca=%d, n_jobs=%s",
            N_total, d_cca, self.n_jobs,
        )

        # Dispatch all N_total folds to the loky process pool.
        # Each worker receives read-only numpy arrays (zero-copy via mmap).
        results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=5)(
            delayed(_validate_single_sample)(v, L_x1, L_y1)
            for v in range(N_total)
        )

        # Reconstruct full cross-validated matrices from fold outputs.
        Z_x = np.empty((N_total, d_cca), dtype=np.float64)
        Z_y = np.empty((N_total, d_cca), dtype=np.float64)
        for v, Z_x_v, Z_y_v in results:
            Z_x[v] = Z_x_v
            Z_y[v] = Z_y_v

        log.info("  Z_x shape: %s | Z_y shape: %s", Z_x.shape, Z_y.shape)
        return Z_x, Z_y

    # -------------------------------------------------------------------------
    # Alignment Error Diagnostics
    # -------------------------------------------------------------------------
    @staticmethod
    def _frobenius_cov_error(
        A: np.ndarray,  # (N, d)
        B: np.ndarray,  # (N, d)
    ) -> float:
        """
        Compute || cov(A.T) - cov(B.T) ||_F^2.

        Uses np.cov which expects (features, observations) layout, hence .T.
        Cholesky is not applicable to a difference matrix (may be indefinite),
        so we use the Frobenius norm directly via vectorised operations.
        """
        diff = np.cov(A.T) - np.cov(B.T)  # (d, d)
        # ||M||_F^2 = trace(M.T @ M) — computed without forming a Python loop.
        return float(np.trace(diff.T @ diff))

    # -------------------------------------------------------------------------
    # Public fit() method — orchestrates the full pipeline
    # -------------------------------------------------------------------------
    def fit(
        self,
        X_shared: np.ndarray,  # (n_x, m_shared)
        Y_shared: np.ndarray,  # (n_y, m_shared)
    ) -> "MorphologicalCoregistrationEngine":
        """
        Execute the full Model-1 pipeline.

        Parameters
        ----------
        X_shared : Core alignment matrix, shape (n_x, m_shared).
        Y_shared : Matching baseline matrix, shape (n_y, m_shared).

        Returns
        -------
        self  (for method chaining)
        """
        # --- Dimension guard -------------------------------------------------
        if X_shared.ndim != 2 or Y_shared.ndim != 2:
            raise ValueError("X_shared and Y_shared must be 2-D arrays.")
        if X_shared.shape[1] != Y_shared.shape[1]:
            raise ValueError(
                f"Feature dimension mismatch: X has {X_shared.shape[1]} "
                f"features, Y has {Y_shared.shape[1]}. m_shared must match."
            )

        n_x, m_shared = X_shared.shape
        n_y           = Y_shared.shape[0]
        N_total       = n_x + n_y
        log.info(
            "fit() called: n_x=%d, n_y=%d, m_shared=%d, N_total=%d",
            n_x, n_y, m_shared, N_total,
        )

        try:
            # Step A
            C_x_0, C_y_0 = self._step_a_gmm_clustering(X_shared, Y_shared)

            # Step B
            self.L_x1, self.L_y1 = self._step_b_lda_projection(
                X_shared, Y_shared, C_x_0, C_y_0
            )

            # Step C
            self.Z_x, self.Z_y = self._step_c_loocv_cca(self.L_x1, self.L_y1)

            # --- In-sample reference covariance error (E_train) --------------
            # Use full LDA spaces projected by a single in-sample CCA fit
            # as the training reference for error comparison.
            d_cca = min(self.L_x1.shape[1], self.L_y1.shape[1])
            cca_full = CCA(n_components=d_cca, max_iter=1000)
            cca_full.fit(self.L_x1, self.L_y1)
            Z_x_train = self.L_x1 @ cca_full.x_rotations_  # (N_total, d_cca)
            Z_y_train = self.L_y1 @ cca_full.y_rotations_  # (N_total, d_cca)

            # E_train = || cov(Z_x_train.T) - cov(Z_y_train.T) ||_F^2
            self.E_train = self._frobenius_cov_error(Z_x_train, Z_y_train)

            # E_val   = || cov(Z_x.T) - cov(Z_y.T) ||_F^2  (out-of-sample)
            self.E_val = self._frobenius_cov_error(self.Z_x, self.Z_y)

            log.info("  E_train (Frobenius^2): %.6f", self.E_train)
            log.info("  E_val   (Frobenius^2): %.6f", self.E_val)

            # --- Construct Z_morph as row-mean of the two canonical views ----
            # Z_morph : (N_total, d_cca) — ground-truth anchor space
            self.Z_morph = 0.5 * (self.Z_x + self.Z_y)
            log.info("  Z_morph shape: %s", self.Z_morph.shape)

        except Exception as exc:
            log.exception("Pipeline failed during fit(): %s", exc)
            raise

        return self

    # -------------------------------------------------------------------------
    # Step D — Inter-Repository Output: save Z_morph as .npy
    # -------------------------------------------------------------------------
    def save_anchor_space(self, filename: str = "method1_optical_anchor_space.npy") -> Path:
        """
        Persist the converged cross-validated latent space Z_morph to disk.

        Output contract (inter-repository boundary)
        -------------------------------------------
        File    : <output_dir>/method1_optical_anchor_space.npy
        Content : numpy binary containing Z_morph of shape (N_total, d_cca).
                  Consumed by TopoUMAP as the 'Z_coordinates' entry of the
                  standardised handoff tuple:
                      (Z_coordinates, cluster_vectors, sample_distance_matrices)

        Returns
        -------
        Path to the written file.
        """
        if self.Z_morph is None:
            raise RuntimeError("Z_morph is None — call fit() before save_anchor_space().")

        out_path = self.output_dir / filename
        np.save(out_path, self.Z_morph)
        log.info("Saved Z_morph %s → %s", self.Z_morph.shape, out_path)
        return out_path

    # -------------------------------------------------------------------------
    # Convenience summary
    # -------------------------------------------------------------------------
    def summary(self) -> dict:
        """Return a flat dict of key pipeline diagnostics."""
        return {
            "Z_morph_shape": None if self.Z_morph is None else self.Z_morph.shape,
            "E_train":       self.E_train,
            "E_val":         self.E_val,
            "k_x":           self.k_x,
            "k_y":           self.k_y,
        }


# =============================================================================
# CLI entry-point — smoke-test with synthetic data
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model-1 Morphological Coregistration Engine")
    parser.add_argument("--n_x",       type=int,   default=120,  help="Samples in Dataset X")
    parser.add_argument("--n_y",       type=int,   default=95,   help="Samples in Dataset Y")
    parser.add_argument("--m_shared",  type=int,   default=40,   help="Shared feature dimension")
    parser.add_argument("--k_x",       type=int,   default=5,    help="GMM/LDA components for X")
    parser.add_argument("--k_y",       type=int,   default=4,    help="GMM/LDA components for Y")
    parser.add_argument("--seed",      type=int,   default=0,    help="NumPy RNG seed")
    parser.add_argument("--output_dir",type=str,   default="outputs")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_shared = rng.standard_normal((args.n_x, args.m_shared))
    Y_shared = rng.standard_normal((args.n_y, args.m_shared))

    engine = MorphologicalCoregistrationEngine(
        k_x=args.k_x,
        k_y=args.k_y,
        gmm_seed=args.seed,
        n_jobs=-1,
        output_dir=args.output_dir,
    )
    engine.fit(X_shared, Y_shared)
    engine.save_anchor_space()
    print(engine.summary())
