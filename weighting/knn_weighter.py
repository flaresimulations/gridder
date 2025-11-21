"""k-NN reweighting of a biased sample to match a parent population.

This script reads two HDF5 files:

* A "parent" (global) population.
* A "sample" population (which may be biased).

The set of kernels is defined by the sample file: every group
``Grids/Kernel_*`` in the sample must also exist in the parent. Each such
group must contain a dataset ``GridPointOverDensities``.

For each kernel:

* Read overdensities from parent and sample.
* Transform as ``log10(1 + delta)``.
* Stack over all kernels into a multi-dimensional feature space.

We then use k-NN density-ratio estimation to compute weights for the sample
so that its multi-scale environment distribution matches the parent.

An optional auto-tuning step searches over candidate k values, choosing the
one that:

* Satisfies an effective sample size (ESS) floor, if possible.
* Minimizes the mean 1D KS distance between parent and weighted sample
  across all dimensions.

Example
-------
    python knn_weighter.py \\
        --parent parent.hdf5 \\
        --sample sample.hdf5 \\
        --autotune_k 35,50,80,100 \\
        --ess_floor 0.20
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def effective_sample_size(w: np.ndarray) -> float:
    """Compute the effective sample size (ESS) of a weight vector.

    ESS is defined as:

    .. math::

        \\text{ESS} = \\frac{(\\sum_i w_i)^2}{\\sum_i w_i^2}.

    Args:
        w: One-dimensional array of non-negative weights.

    Returns:
        Effective sample size as a float.
    """
    s = float(w.sum())
    return (s * s) / float(np.sum(w * w) + 1e-18)


def weighted_ks(
    x_parent: np.ndarray,
    x_sample: np.ndarray,
    w_sample: np.ndarray,
    rng: np.random.Generator,
    n_parent: int = 100_000,
    n_sample: int = 100_000,
) -> float:
    """Compute a weighted 1D Kolmogorov-Smirnov distance.

    The parent distribution is unweighted; the sample distribution is
    weighted. For speed, both are optionally downsampled before computing
    the KS distance.

    Args:
        x_parent: Parent data values (1D).
        x_sample: Sample data values (1D).
        w_sample: Weights for ``x_sample``, same length as ``x_sample``.
        rng: NumPy random number generator instance.
        n_parent: Maximum number of parent points to use.
        n_sample: Maximum number of sample points to use.

    Returns:
        The KS distance as a float.
    """
    # Downsample parent.
    if len(x_parent) > n_parent:
        xp = x_parent[rng.choice(len(x_parent), n_parent, replace=False)]
    else:
        xp = x_parent

    # Downsample sample + weights.
    if len(x_sample) > n_sample:
        idx = rng.choice(len(x_sample), n_sample, replace=False)
        xs = x_sample[idx]
        ws = w_sample[idx]
    else:
        xs, ws = x_sample, w_sample

    # Sort and build CDFs.
    xp = np.sort(xp)
    order = np.argsort(xs)
    xs = xs[order]
    ws = np.clip(ws[order], 0, None)
    ws_sum = ws.sum()
    if ws_sum < 1e-15:
        # All weights are effectively zero - return worst-case KS distance
        return 1.0
    ws /= ws_sum
    cdf_s = np.cumsum(ws)

    grid = np.unique(np.concatenate([xp, xs]))
    cdf_p = np.searchsorted(xp, grid, side="right") / len(xp)

    idx = np.searchsorted(xs, grid, side="right") - 1
    idx = np.clip(idx, -1, len(xs) - 1)
    cdf_s_at_grid = np.where(idx >= 0, cdf_s[idx], 0.0)

    return float(np.max(np.abs(cdf_p - cdf_s_at_grid)))


def whiten_by_parent(
    Xp: np.ndarray,
    Xs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply parent-covariance whitening (Mahalanobis transform).

    The parent distribution defines the mean and covariance used to whiten
    both the parent and the sample data:

    .. math::

        X' = (X - \\mu_\\text{parent}) L^{-1},

    where ``L`` is the Cholesky factor of the parent covariance.

    Args:
        Xp: Parent data array of shape ``(N, D)``.
        Xs: Sample data array of shape ``(M, D)``.

    Returns:
        Tuple ``(Xp_whitened, Xs_whitened)`` with the same shapes as inputs.
    """
    mu = Xp.mean(axis=0)
    Xm = Xp - mu
    Sigma = np.cov(Xm, rowvar=False)
    d = Sigma.shape[0]
    # Tiny ridge for numerical stability.
    Sigma = Sigma + (1e-12 * np.trace(Sigma) / max(d, 1)) * np.eye(d)
    L = np.linalg.cholesky(Sigma)
    Linv = np.linalg.inv(L)
    return (Xp - mu) @ Linv, (Xs - mu) @ Linv


def filter_and_transform_overdensities(
    overdens: Dict[float, np.ndarray],
) -> Tuple[Dict[float, np.ndarray], np.ndarray]:
    """Filter out -1 values and apply log10(1 + delta) transform.

    Grid points with overdensity = -1 indicate no particles nearby.
    These must be removed before applying the log transform to avoid NaN.

    Args:
        overdens: Mapping from scale to raw overdensity arrays (may contain -1).

    Returns:
        Tuple (transformed_overdens, mask) where:
            - transformed_overdens: log10(1 + delta) for valid points
            - mask: Boolean array indicating which rows were kept

    Raises:
        ValueError: If arrays have mismatched lengths.
    """
    if not overdens:
        return overdens, np.array([], dtype=bool)

    first = next(iter(overdens.values()))
    length = len(first)

    # Build mask: keep rows where ALL scales have delta > -1
    mask = np.ones(length, dtype=bool)
    for scale, arr in overdens.items():
        if len(arr) != length:
            raise ValueError(
                "All arrays must have the same length; found mismatched lengths."
            )
        mask &= arr > -1.0

    n_dropped = length - np.sum(mask)
    if n_dropped > 0:
        print(
            f"[Filter] Dropped {n_dropped:,}/{length:,} "
            f"({100.0 * n_dropped / length:.2f}%) grid points with delta <= -1 "
            f"(no particles nearby)."
        )

    # Apply log transform to filtered data
    transformed = {scale: np.log10(arr[mask] + 1.0) for scale, arr in overdens.items()}

    return transformed, mask


def drop_nan_rows_per_scale(
    samples: Dict[float, np.ndarray],
) -> Dict[float, np.ndarray]:
    """Remove entries that are NaN in any dimension for a population.

    For a dictionary of 1D arrays (one per smoothing scale), this function
    removes indices where *any* scale is NaN, ensuring consistent length
    across all scales.

    Args:
        samples: Mapping from scale (e.g., in Mpc/h) to 1D NumPy arrays of
            equal length.

    Returns:
        New dictionary with NaN-containing entries removed consistently
        across all scales.

    Raises:
        ValueError: If the arrays in ``samples`` do not all have the same
            length.
    """
    if not samples:
        return samples

    first = next(iter(samples.values()))
    length = len(first)
    mask = np.ones(length, dtype=bool)

    for arr in samples.values():
        if len(arr) != length:
            raise ValueError(
                "All arrays in a population must have the same length; "
                "found mismatched lengths."
            )
        mask &= ~np.isnan(arr)

    n_dropped = length - np.sum(mask)
    if n_dropped > 0:
        print(
            f"[NaN] Dropped {n_dropped:,}/{length:,} "
            f"({100.0 * n_dropped / length:.2f}%) entries containing NaN."
        )

    return {scale: arr[mask] for scale, arr in samples.items()}


# ---------------------------------------------------------------------
# Core k-NN reweighter with auto-tuning
# ---------------------------------------------------------------------


@dataclass
class KNNReweighterConfig:
    """Configuration for :class:`KNNReweighter`.

    Attributes:
        k: Default number of neighbors for k-NN if auto-tuning is disabled.
        autotune_k: Candidate k values to try when auto-tuning. If None,
            a single fixed k is used.
        ess_floor: Minimum ESS fraction (ESS / M) for a candidate k to be
            considered "feasible" during auto-tuning.
        preprocess: Preprocessing mode: ``"whiten"``, ``"standardize"``, or
            ``"none"``.
        standardize_by_parent: If True and ``preprocess == "standardize"``,
            use parent statistics only; otherwise, use pooled stats.
        algorithm: Nearest-neighbors algorithm (as in scikit-learn).
        metric: Distance metric (e.g., ``"minkowski"``, ``"euclidean"``).
        leaf_size: Leaf size for tree-based nearest-neighbor algorithms.
        tau: Temperature exponent applied to the weights (``w := w**tau``).
            Use tau < 1 to reduce weight variance (smoother, higher ESS).
            Use tau > 1 to increase contrast (sharper weights, lower ESS).
            Default tau=1 preserves density ratio exactly.
        clip_range: Tuple ``(min, max)`` for clipping weights.
        ks_eval_parent: Maximum number of parent points for KS evaluation.
        ks_eval_sample: Maximum number of sample points for KS evaluation.
        random_state: Random seed for diagnostics and KS subsampling.
    """

    # k / tuning
    k: int = 50
    autotune_k: Optional[List[int]] = None
    ess_floor: float = 0.20  # as fraction of M

    # preprocessing
    preprocess: str = "whiten"  # whiten | standardize | none
    standardize_by_parent: bool = True  # only used if preprocess=standardize

    # neighbors
    algorithm: str = "auto"
    metric: str = "minkowski"
    leaf_size: int = 40

    # stabilization
    tau: float = 1.0  # tau < 1: reduce variance, tau > 1: increase contrast
    clip_range: Tuple[float, float] = (0.01, 50.0)

    # diagnostics
    ks_eval_parent: int = 100_000
    ks_eval_sample: int = 100_000
    random_state: int = 42


class KNNReweighter:
    """k-NN density-ratio reweighter for multi-scale environments."""

    def __init__(self, cfg: KNNReweighterConfig):
        """Initialize the reweighter.

        Args:
            cfg: Configuration object controlling k-NN and diagnostics.
        """
        self.cfg = cfg
        self.scales: List[float] = []
        self.weights_: Optional[np.ndarray] = None
        self.info_: Optional[Dict] = None

    def fit(
        self,
        parent_samples: Dict[float, np.ndarray],
        sample_samples: Dict[float, np.ndarray],
    ) -> Tuple[np.ndarray, Dict]:
        """Compute weights that map the sample to the parent distribution.

        The features are constructed by column-stacking the overdensities at
        different smoothing scales.

        Args:
            parent_samples: Mapping from scale to parent overdensity array
                (1D, same length per scale).
            sample_samples: Mapping from scale to sample overdensity array
                (1D, same length per scale).

        Returns:
            Tuple ``(weights, info)``:
                * ``weights``: 1D array of length M with reweighting factors.
                * ``info``: Dictionary of diagnostics and configuration values.

        Raises:
            ValueError: If the parent and sample do not share the same set
                of scales.
        """
        if set(parent_samples.keys()) != set(sample_samples.keys()):
            raise ValueError(
                "parent_samples and sample_samples must have identical keys "
                "(same set of smoothing scales)."
            )

        self.scales = sorted(parent_samples.keys())
        Xp = np.column_stack([parent_samples[s] for s in self.scales])  # [N, D]
        Xs = np.column_stack([sample_samples[s] for s in self.scales])  # [M, D]
        N, d = Xp.shape
        M = Xs.shape[0]
        rng = np.random.default_rng(self.cfg.random_state)

        print(f"[kNN] {d}D features | Parent N={N:,} | Sample M={M:,}")
        Xp_prep, Xs_prep = self._preprocess(Xp, Xs)

        # Solve for weights, optionally auto-tuning k.
        if self.cfg.autotune_k:
            best_k, w, tuning_info = self._autotune_k(
                Xp_prep, Xs_prep, Xp, Xs, N, M, d, rng
            )
            k_used = best_k
        else:
            k_used = self.cfg.k
            w = self._weights_for_k(Xp_prep, Xs_prep, N, M, d, k_used)
            tuning_info = None

        # Basic diagnostics: ESS, clipping, KS.
        ess = effective_sample_size(w)
        clip_frac = float(
            np.mean(
                (w <= self.cfg.clip_range[0] + 1e-15)
                | (w >= self.cfg.clip_range[1] - 1e-15)
            )
        )

        ks_vals = [
            weighted_ks(
                Xp[:, j],
                Xs[:, j],
                w,
                rng,
                n_parent=min(self.cfg.ks_eval_parent, N),
                n_sample=min(self.cfg.ks_eval_sample, M),
            )
            for j in range(d)
        ]
        ks_mean = float(np.mean(ks_vals))

        info = {
            "method": "knn_density_ratio",
            "k": int(k_used),
            "ks_mean": ks_mean,
            "ks_per_dim": ks_vals,
            "preprocess": self.cfg.preprocess,
            "standardize_by_parent": self.cfg.standardize_by_parent,
            "scales": self.scales,
            "parent_samples": N,
            "sample_samples": M,
            "algorithm": self.cfg.algorithm,
            "metric": self.cfg.metric,
            "leaf_size": self.cfg.leaf_size,
            "tau": self.cfg.tau,
            "clip_range": self.cfg.clip_range,
            "clip_fraction": clip_frac,
            "ess": float(ess),
            "mean_weight": float(np.mean(w)),
            "std_weight": float(np.std(w)),
            "min_weight": float(np.min(w)),
            "max_weight": float(np.max(w)),
            "tuning": tuning_info,
        }

        print(
            f"[kNN] k={k_used} | KS(mean)={ks_mean:.4f} | "
            f"ESS={ess:,.0f}/{M:,} | clipped={100 * clip_frac:.1f}%"
        )

        self.weights_ = w
        self.info_ = info
        return w, info

    # --------------------- internals ---------------------

    def _preprocess(
        self,
        Xp: np.ndarray,
        Xs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the requested preprocessing to parent and sample data.

        Args:
            Xp: Parent data array of shape ``(N, D)``.
            Xs: Sample data array of shape ``(M, D)``.

        Returns:
            Tuple ``(Xp_processed, Xs_processed)``.

        Raises:
            ValueError: If ``preprocess`` is not one of the supported modes.
        """
        mode = self.cfg.preprocess.lower()

        if mode == "whiten":
            print("[kNN] Preprocess: parent-covariance whitening.")
            return whiten_by_parent(Xp, Xs)

        if mode == "standardize":
            if self.cfg.standardize_by_parent:
                mu = Xp.mean(axis=0)
                sd = Xp.std(axis=0) + 1e-12
                src = "parent"
            else:
                both = np.vstack([Xp, Xs])
                mu = both.mean(axis=0)
                sd = both.std(axis=0) + 1e-12
                src = "pooled"
            print(f"[kNN] Preprocess: z-score (by {src} stats).")
            return (Xp - mu) / sd, (Xs - mu) / sd

        if mode == "none":
            print("[kNN] Preprocess: none.")
            return Xp, Xs

        raise ValueError('preprocess must be "whiten", "standardize", or "none"')

    def _knn_dists(
        self,
        X_query: np.ndarray,
        X_ref: np.ndarray,
        *,
        k: int,
    ) -> np.ndarray:
        """Compute distances to the k nearest neighbors.

        Args:
            X_query: Query points of shape ``(Q, D)``.
            X_ref: Reference points of shape ``(R, D)``.
            k: Number of neighbors to retrieve.

        Returns:
            Array of shape ``(Q, k)`` with distances to the k nearest neighbors.
        """
        nn = NearestNeighbors(
            n_neighbors=k,
            algorithm=self.cfg.algorithm,
            leaf_size=self.cfg.leaf_size,
            metric=self.cfg.metric,
        ).fit(X_ref)
        return nn.kneighbors(X_query, return_distance=True)[0]

    def _weights_for_k(
        self,
        Xp_prep: np.ndarray,
        Xs_prep: np.ndarray,
        N: int,
        M: int,
        d: int,
        k: int,
    ) -> np.ndarray:
        """Compute stabilized weights for a single k.

        The core k-NN density-ratio formula is:

        .. math::

            w_i \\propto \\frac{M}{N}\\left(\\frac{r_\\text{sample, i}}
            {r_\\text{parent, i}}\\right)^d,

        with optional temperature exponent, clipping, and renormalization.

        Args:
            Xp_prep: Preprocessed parent data, shape ``(N, D)``.
            Xs_prep: Preprocessed sample data, shape ``(M, D)``.
            N: Number of parent points.
            M: Number of sample points.
            d: Dimensionality of the feature space.
            k: Number of neighbors.

        Returns:
            One-dimensional array of weights of length M.

        Raises:
            ValueError: If k is out of valid range.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got k={k}")
        if k >= N:
            raise ValueError(f"k must be < N (parent size), got k={k}, N={N}")
        if k + 1 > M:
            raise ValueError(f"k+1 must be <= M (sample size), got k={k}, M={M}")

        dist_p = self._knn_dists(Xs_prep, Xp_prep, k=k)  # [M, k]
        r_parent = dist_p[:, -1]

        dist_s = self._knn_dists(Xs_prep, Xs_prep, k=k + 1)  # include self
        r_sample = dist_s[:, -1]

        eps = 1e-12
        w = (M / N) * ((r_sample + eps) / (r_parent + eps)) ** d

        if self.cfg.tau != 1.0:
            w = np.power(w, self.cfg.tau)

        w = np.clip(w, self.cfg.clip_range[0], self.cfg.clip_range[1])
        w *= len(w) / (w.sum() + 1e-18)
        return w

    def _autotune_k(
        self,
        Xp_prep: np.ndarray,
        Xs_prep: np.ndarray,
        Xp_raw: np.ndarray,
        Xs_raw: np.ndarray,
        N: int,
        M: int,
        d: int,
        rng: np.random.Generator,
    ) -> Tuple[int, np.ndarray, Dict]:
        """Grid-search over candidate k values with an ESS floor.

        Args:
            Xp_prep: Preprocessed parent data, shape ``(N, D)``.
            Xs_prep: Preprocessed sample data, shape ``(M, D)``.
            Xp_raw: Raw parent data for KS diagnostics, shape ``(N, D)``.
            Xs_raw: Raw sample data for KS diagnostics, shape ``(M, D)``.
            N: Number of parent points.
            M: Number of sample points.
            d: Dimensionality of the feature space.
            rng: NumPy random generator for KS subsampling.

        Returns:
            Tuple ``(best_k, best_weights, summary_dict)`` where
            ``summary_dict`` contains per-k diagnostics.
        """
        candidates = sorted(
            {int(k) for k in self.cfg.autotune_k if 1 <= int(k) < min(N, M)}
        )
        if not candidates:
            raise ValueError(
                "autotune_k is non-empty but no valid candidates remain. "
                "Check that k < min(N, M)."
            )

        print(
            f"[kNN] Auto-tune k over {candidates} with ESS floor "
            f"{int(self.cfg.ess_floor * 100)}%..."
        )

        results = []
        best_tuple = None
        best_k = None
        best_w = None

        for k in candidates:
            w = self._weights_for_k(Xp_prep, Xs_prep, N, M, d, k)
            ess = effective_sample_size(w)

            ks_vals = [
                weighted_ks(
                    Xp_raw[:, j],
                    Xs_raw[:, j],
                    w,
                    rng,
                    n_parent=min(self.cfg.ks_eval_parent, N),
                    n_sample=min(self.cfg.ks_eval_sample, M),
                )
                for j in range(d)
            ]
            ks_mean = float(np.mean(ks_vals))
            feasible = ess >= self.cfg.ess_floor * M
            results.append(
                {"k": k, "ess": float(ess), "ks_mean": ks_mean, "feasible": feasible}
            )

            # Selection criteria (lexicographic ordering):
            # 1. Prefer feasible (ESS >= floor) over infeasible
            # 2. Among same feasibility, prefer lower KS distance
            # 3. Among same KS, prefer higher ESS
            # 4. Among same ESS, prefer smaller k (simpler model)
            selection_key = (not feasible, ks_mean, -ess, k)
            if (best_tuple is None) or (selection_key < best_tuple):
                best_tuple, best_k, best_w = selection_key, k, w

        print(
            "[kNN] Tuning results: "
            + ", ".join(
                f"k={r['k']}: KS={r['ks_mean']:.3f}, ESS={int(r['ess'])}"
                + ("" if r["feasible"] else "")
                for r in results
            )
        )

        summary = {
            "candidates": candidates,
            "results": results,
            "best_k": best_k,
            "ess_floor": self.cfg.ess_floor,
        }
        return best_k, best_w, summary


# ---------------------------------------------------------------------
# HDF5 loading
# ---------------------------------------------------------------------


def load_multiscale_overdensities(
    parent_path: str,
    sample_path: str,
) -> Tuple[Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """Load raw overdensities from parent and sample HDF5 files.

    The set of kernels is defined by the *sample* file. For every group
    ``Grids/Kernel_N`` present in the sample file:

    * The same group must exist in the parent file.
    * Each group must contain a dataset ``GridPointOverDensities``.
    * The kernel radius is read from the group's ``KernelRadius`` attribute.
    * Raw overdensity values are returned (NOT log-transformed).

    Note: Values of -1 indicate grid points with no particles nearby.
    Use ``filter_and_transform_overdensities()`` to remove these and
    apply the log10(1 + delta) transform.

    The returned dictionaries map the smoothing scale (from the KernelRadius
    attribute) to a 1D NumPy array of raw overdensities.

    Args:
        parent_path: Path to the parent/global population HDF5 file.
        sample_path: Path to the sample/observed population HDF5 file.

    Returns:
        Tuple ``(parent_data, sample_data)``, where each is a mapping from
        scale to 1D NumPy arrays.

    Raises:
        KeyError: If required groups/datasets are missing.
        ValueError: If no kernels are found; if KernelRadius attribute is
            missing; or if the parent is missing kernels found in the sample
            file.
    """
    parent_data: Dict[float, np.ndarray] = {}
    sample_data: Dict[float, np.ndarray] = {}

    with (
        h5py.File(sample_path, "r") as f_sample,
        h5py.File(parent_path, "r") as f_parent,
    ):
        if "Grids" not in f_sample:
            raise KeyError("Sample file is missing group 'Grids'.")
        if "Grids" not in f_parent:
            raise KeyError("Parent file is missing group 'Grids'.")

        sample_grids = f_sample["Grids"]
        parent_grids = f_parent["Grids"]

        kernel_keys = [k for k in sample_grids.keys() if k.startswith("Kernel_")]
        if not kernel_keys:
            raise ValueError(
                "No kernel groups found in sample file under 'Grids'. "
                "Expected groups named like 'Kernel_0', 'Kernel_1', etc."
            )

        missing = [k for k in kernel_keys if k not in parent_grids]
        if missing:
            raise ValueError(
                "Parent file is missing the following kernel groups "
                f"present in the sample file: {', '.join(missing)}"
            )

        print(
            f"[HDF5] Found {len(kernel_keys)} kernels in sample file: "
            + ", ".join(kernel_keys)
        )

        for key in kernel_keys:
            # Read kernel radius from group attribute
            try:
                scale = float(sample_grids[key].attrs["KernelRadius"])
            except KeyError as exc:
                raise ValueError(
                    f"Kernel group '{key}' is missing 'KernelRadius' attribute. "
                    "Cannot determine smoothing scale."
                ) from exc

            try:
                parent_ds = parent_grids[key]["GridPointOverDensities"][...]
                sample_ds = sample_grids[key]["GridPointOverDensities"][...]
            except KeyError as exc:
                raise KeyError(
                    f"Missing 'GridPointOverDensities' for kernel '{key}'."
                ) from exc

            # Store raw overdensities (will be filtered and transformed later)
            # Note: -1 values indicate grid points with no particles nearby
            parent_data[scale] = np.asarray(parent_ds, dtype=float)
            sample_data[scale] = np.asarray(sample_ds, dtype=float)

    return parent_data, sample_data


# ---------------------------------------------------------------------
# Plotting and simple diagnostics
# ---------------------------------------------------------------------


def plot_diagnostics(
    parent_samples: Dict[float, np.ndarray],
    sample_samples: Dict[float, np.ndarray],
    weights: np.ndarray,
    info: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot simple diagnostics: 1D distributions and weight histogram.

    The plot has three panels:

    * Parent vs sample vs weighted sample for the first scale.
    * Parent vs sample vs weighted sample for the second scale (if present).
    * Histogram of the weights.

    Args:
        parent_samples: Mapping from scale to parent overdensity array.
        sample_samples: Mapping from scale to sample overdensity array.
        weights: Weights for the sample population.
        info: Diagnostics dictionary returned by :meth:`KNNReweighter.fit`.
        save_path: Optional path to save the diagnostic PNG.

    Returns:
        The Matplotlib Figure instance.
    """
    scales = sorted(parent_samples.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes = axes.flatten()

    def _plot_1d(ax, scale: float) -> None:
        parent = parent_samples[scale]
        sample = sample_samples[scale]
        bins = np.linspace(
            min(parent.min(), sample.min()), max(parent.max(), sample.max()), 50
        )
        ax.hist(
            parent,
            bins=bins,
            density=True,
            label="Parent",
            histtype="step",
            lw=2,
        )
        ax.hist(
            sample,
            bins=bins,
            density=True,
            label="Sample (raw)",
            histtype="step",
            lw=2,
        )
        ax.hist(
            sample,
            bins=bins,
            weights=weights,
            density=True,
            label="Sample (weighted)",
            histtype="step",
            lw=2,
        )
        ax.set_xlabel(f"� (R={scale} Mpc/h)")
        ax.set_ylabel("PDF")
        ax.set_title(f"Scale {scale} Mpc/h")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # First scale.
    _plot_1d(axes[0], scales[0])

    # Second scale (if available).
    if len(scales) > 1:
        _plot_1d(axes[1], scales[1])
    else:
        axes[1].axis("off")

    # Weight histogram.
    axw = axes[2]
    axw.hist(weights, bins=60, alpha=0.85, edgecolor="black")
    axw.axvline(
        np.mean(weights),
        color="red",
        linestyle="--",
        lw=2,
        label=f"Mean={np.mean(weights):.3f}",
    )
    axw.set_xlabel("Weight")
    axw.set_ylabel("Count")
    ess = int(info["ess"])
    M = info["sample_samples"]
    ks_mean = info["ks_mean"]
    axw.set_title(f"w distribution | ESS={ess}/{M:,}, KS(mean)={ks_mean:.3f}")
    axw.legend()
    axw.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[plot] Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run_full_analysis(
    parent_samples: Dict[float, np.ndarray],
    sample_samples: Dict[float, np.ndarray],
    show_plots: bool = True,
    save_path: Optional[str] = None,
    **knn_kwargs,
) -> Tuple[np.ndarray, Dict]:
    """Run the full k-NN reweighting and simple diagnostics.

    Args:
        parent_samples: Mapping from scale to parent overdensity array.
        sample_samples: Mapping from scale to sample overdensity array.
        show_plots: If True, display the diagnostic plot.
        save_path: Optional path to save the diagnostic PNG.
        **knn_kwargs: Keyword arguments used to configure
            :class:`KNNReweighterConfig`.

    Returns:
        Tuple ``(weights, info)``:
            * ``weights``: 1D array of weights for the sample population.
            * ``info``: Diagnostics dictionary from
              :meth:`KNNReweighter.fit`.
    """
    print("=== Multi-Scale Environment Weighting (k-NN, auto-k) ===")

    cfg = KNNReweighterConfig(**knn_kwargs)
    rw = KNNReweighter(cfg)

    weights, info = rw.fit(parent_samples, sample_samples)

    t = info.get("tuning")
    tune_str = ""
    if t:
        tune_str = (
            f" | auto-k over {t['candidates']} "
            f"-> best={t['best_k']} (ESS floor {int(t['ess_floor'] * 100)}%)"
        )

    print(
        "\n[Summary] "
        f"k={info['k']}, KS(mean)={info['ks_mean']:.4f}, "
        f"ESS={int(info['ess'])}/{info['sample_samples']}, "
        f"clip={info['clip_range']}, "
        f"clipped={100 * info['clip_fraction']:.1f}%"
        f"{tune_str}"
    )

    if show_plots:
        print("\n[Plot] Rendering diagnostics...")
        _ = plot_diagnostics(parent_samples, sample_samples, weights, info, save_path)
        plt.show()

    return weights, info


# ---------------------------------------------------------------------
# __main__ (argparse)
# ---------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "k-NN density-ratio weighting for multi-scale environments.\n\n"
            "The sample file defines which Grids/Kernel_* groups are used; "
            "the same kernels must exist in the parent file."
        )
    )

    # Required I/O.
    parser.add_argument(
        "--parent",
        type=str,
        required=True,
        help="Path to parent/global population HDF5 file.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Path to sample/observed population HDF5 file.",
    )

    # k / auto-tuning.
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Default number of neighbors (used if --autotune_k is empty).",
    )
    parser.add_argument(
        "--autotune_k",
        type=str,
        default="",
        help=(
            "Comma-separated candidate k values for auto-tuning, "
            "e.g. '35,50,80,100'. If empty, k is used directly."
        ),
    )
    parser.add_argument(
        "--ess_floor",
        type=float,
        default=0.20,
        help="Minimum ESS fraction (ESS / M) for a candidate k to be feasible.",
    )

    # Preprocessing.
    parser.add_argument(
        "--preprocess",
        choices=["whiten", "standardize", "none"],
        default="whiten",
        help="Feature preprocessing mode.",
    )
    parser.add_argument(
        "--standardize_by_parent",
        action="store_true",
        help="If preprocess=standardize, use parent stats (default True).",
    )
    parser.add_argument(
        "--standardize_by_pooled",
        dest="standardize_by_parent",
        action="store_false",
        help="If preprocess=standardize, use pooled parent+sample stats.",
    )
    parser.set_defaults(standardize_by_parent=True)

    # Neighbors.
    parser.add_argument(
        "--metric",
        type=str,
        default="minkowski",
        help="Distance metric (e.g., 'minkowski', 'euclidean').",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="auto",
        help="NearestNeighbors algorithm (scikit-learn).",
    )
    parser.add_argument(
        "--leaf_size",
        type=int,
        default=40,
        help="Leaf size for tree-based nearest-neighbor algorithms.",
    )

    # Stabilization.
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature exponent (weights := weights ** tau).",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=0.01,
        help="Lower clip bound for weights.",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=50.0,
        help="Upper clip bound for weights.",
    )

    # Misc / plotting.
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for diagnostics and KS subsampling.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save diagnostic PNG.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable plotting.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save computed weights as .npy file.",
    )

    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load data from the two HDF5 files.
    parent_data, sample_data = load_multiscale_overdensities(
        parent_path=args.parent,
        sample_path=args.sample,
    )

    # Remove NaNs consistently within each population.
    parent_data = drop_nan_rows_per_scale(parent_data)
    sample_data = drop_nan_rows_per_scale(sample_data)

    # Build kwargs for run_full_analysis.
    auto_list = [int(x) for x in args.autotune_k.split(",") if x.strip().isdigit()]
    kwargs = dict(
        k=args.k,
        autotune_k=auto_list if auto_list else None,
        ess_floor=args.ess_floor,
        preprocess=args.preprocess,
        standardize_by_parent=args.standardize_by_parent,
        algorithm=args.algorithm,
        metric=args.metric,
        leaf_size=args.leaf_size,
        tau=args.tau,
        clip_range=(args.clip_min, args.clip_max),
        random_state=args.random_state,
    )

    # Run pipeline.
    weights, info = run_full_analysis(
        parent_data,
        sample_data,
        show_plots=not args.no_plots,
        save_path=args.save,
        **kwargs,
    )

    # Save weights to file if requested.
    if args.output:
        np.save(args.output, weights)
        print(f"[Output] Saved weights to: {args.output}")

    print(f"\n[Done] Generated {len(weights)} weights.")
