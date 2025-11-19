"""Test k-NN weighting on halo mass functions.

This script evaluates how well k-NN density-ratio weighting can reproduce
the true halo mass function (HMF) from a small subsample.

Workflow:
1. Load multi-scale overdensity features from parent (global) and sample (full box) HDF5 files
2. Load halo masses for the full sample population (the truth)
3. Subsample N halos randomly from the full sample
4. Apply k-NN weighting to the subsample using parent environment as reference
5. Compare: Truth (full sample) vs Subsample (raw) vs Subsample (weighted)

The goal is to test whether a small weighted subsample can reproduce the
true HMF of the full sample.

The script produces diagnostic plots showing:
- HMF comparison: Truth vs raw subsample vs weighted subsample
- Environment distributions at different scales
- Weight distribution statistics

Example
-------
    python test_hmf_weighting.py \\
        --parent parent_grids.hdf5 \\
        --sample sample_grids.hdf5 \\
        --masses sample_halo_masses.txt \\
        --subsample 10000 \\
        --autotune_k 35,50,80,100 \\
        --output weighted_masses.npy \\
        --save hmf_diagnostics.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Import the k-NN weighting machinery
from knn_weighter import (
    KNNReweighter,
    KNNReweighterConfig,
    drop_nan_rows_per_scale,
    effective_sample_size,
    filter_and_transform_overdensities,
    load_multiscale_overdensities,
)


def load_halo_masses(mass_file: str) -> np.ndarray:
    """Load halo masses from a text file.

    Args:
        mass_file: Path to text file containing one mass per line.

    Returns:
        1D array of halo masses.

    Raises:
        ValueError: If file is empty or contains invalid data.
    """
    masses = np.loadtxt(mass_file)
    if masses.ndim != 1:
        raise ValueError(
            f"Mass file should contain a single column, got shape {masses.shape}"
        )
    if len(masses) == 0:
        raise ValueError("Mass file is empty")

    print(f"[Masses] Loaded {len(masses):,} halo masses from {mass_file}")
    print(f"[Masses] Mass range: {masses.min():.2e} - {masses.max():.2e}")

    return masses


def subsample_data(
    sample_overdens: Dict[float, np.ndarray],
    masses: np.ndarray,
    n_subsample: int,
    random_state: int = 42,
) -> Tuple[Dict[float, np.ndarray], np.ndarray, np.ndarray]:
    """Randomly subsample halos and their environment features.

    Args:
        sample_overdens: Mapping from scale to sample overdensity arrays.
        masses: Halo masses (same length as overdensity arrays).
        n_subsample: Number of halos to randomly select.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple (subsampled_overdens, subsampled_masses, indices).

    Raises:
        ValueError: If n_subsample > sample size.
    """
    n_total = len(masses)
    if n_subsample > n_total:
        raise ValueError(
            f"Requested subsample size {n_subsample:,} exceeds "
            f"sample size {n_total:,}"
        )

    rng = np.random.default_rng(random_state)
    indices = rng.choice(n_total, size=n_subsample, replace=False)

    subsampled_overdens = {
        scale: arr[indices] for scale, arr in sample_overdens.items()
    }
    subsampled_masses = masses[indices]

    print(
        f"[Subsample] Randomly selected {n_subsample:,}/{n_total:,} halos "
        f"(seed={random_state})"
    )
    print(
        f"[Subsample] Subsampled mass range: "
        f"{subsampled_masses.min():.2e} - {subsampled_masses.max():.2e}"
    )

    return subsampled_overdens, subsampled_masses, indices


def compute_hmf(
    masses: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_bins: int = 20,
    mass_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute halo mass function (differential number density).

    Args:
        masses: Halo masses.
        weights: Optional weights for each halo.
        n_bins: Number of mass bins.
        mass_range: Optional (min, max) mass range. If None, uses data range.

    Returns:
        Tuple (bin_centers, dn_dlogM, bin_edges) where dn_dlogM is the
        differential number density per logarithmic mass bin.
    """
    if weights is None:
        weights = np.ones_like(masses)

    if mass_range is None:
        mass_range = (masses.min(), masses.max())

    log_masses = np.log10(masses)
    log_range = (np.log10(mass_range[0]), np.log10(mass_range[1]))

    counts, bin_edges = np.histogram(
        log_masses, bins=n_bins, range=log_range, weights=weights
    )
    bin_widths = np.diff(bin_edges)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Differential number density: dn/dlog10(M)
    dn_dlogM = counts / bin_widths
    # Normalize to total number
    dn_dlogM *= len(masses) / dn_dlogM.sum() / bin_widths.mean()

    return bin_centers, dn_dlogM, bin_edges


def plot_hmf_comparison(
    truth_masses: np.ndarray,
    subsample_masses: np.ndarray,
    weights: np.ndarray,
    info: Dict,
    parent_overdens: Dict[float, np.ndarray],
    subsample_overdens: Dict[float, np.ndarray],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot HMF comparison and diagnostics.

    Creates a multi-panel figure with:
    - HMF comparison (truth vs raw subsample vs weighted subsample)
    - Environment distributions at two scales
    - Weight distribution histogram

    Args:
        truth_masses: True halo masses (full sample).
        subsample_masses: Subsampled halo masses.
        weights: Weights for subsampled halos.
        info: Diagnostics from KNNReweighter.
        parent_overdens: Parent overdensity features (global reference).
        subsample_overdens: Subsample overdensity features.
        save_path: Optional path to save figure.

    Returns:
        Figure instance.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    scales = sorted(parent_overdens.keys())

    # Determine common mass range for HMF
    all_masses = np.concatenate([truth_masses, subsample_masses])
    mass_range = (all_masses.min(), all_masses.max())

    # --- Panel 1: Halo Mass Function ---
    ax_hmf = fig.add_subplot(gs[0, :])

    # Compute HMFs
    bc_truth, hmf_truth, _ = compute_hmf(
        truth_masses, n_bins=25, mass_range=mass_range
    )
    bc_subsample, hmf_subsample, _ = compute_hmf(
        subsample_masses, n_bins=25, mass_range=mass_range
    )
    bc_weighted, hmf_weighted, _ = compute_hmf(
        subsample_masses, weights=weights, n_bins=25, mass_range=mass_range
    )

    ax_hmf.plot(
        bc_truth, hmf_truth, "k-", lw=2.5, label=f"Truth (full sample, N={len(truth_masses):,})"
    )
    ax_hmf.plot(
        bc_subsample,
        hmf_subsample,
        "r--",
        lw=2,
        alpha=0.7,
        label=f"Subsample (raw, N={len(subsample_masses):,})",
    )
    ax_hmf.plot(
        bc_weighted,
        hmf_weighted,
        "b-",
        lw=2,
        alpha=0.8,
        label=f"Subsample (weighted, ESS={int(info['ess'])}/{len(subsample_masses):,})",
    )

    ax_hmf.set_xlabel(r"$\log_{10}(M_{\rm halo} \, [M_\odot])$", fontsize=13)
    ax_hmf.set_ylabel(r"$dn / d\log_{10}M$ (normalized)", fontsize=13)
    ax_hmf.set_title(
        f"Halo Mass Function | k={info['k']}, KS(env)={info['ks_mean']:.3f}",
        fontsize=14,
        fontweight="bold",
    )
    ax_hmf.legend(fontsize=11, loc="upper right")
    ax_hmf.grid(True, alpha=0.3)
    ax_hmf.set_yscale("log")

    # --- Panel 2: Environment at scale 0 ---
    ax_env1 = fig.add_subplot(gs[1, 0])
    _plot_environment_dist(
        ax_env1, parent_overdens[scales[0]], sample_overdens[scales[0]], weights, scales[0]
    )

    # --- Panel 3: Environment at scale 1 (if available) ---
    ax_env2 = fig.add_subplot(gs[1, 1])
    if len(scales) > 1:
        _plot_environment_dist(
            ax_env2,
            parent_overdens[scales[1]],
            sample_overdens[scales[1]],
            weights,
            scales[1],
        )
    else:
        ax_env2.text(
            0.5, 0.5, "Only 1 scale available", ha="center", va="center", fontsize=12
        )
        ax_env2.axis("off")

    # --- Panel 4: Weight distribution ---
    ax_weights = fig.add_subplot(gs[1, 2])
    ax_weights.hist(weights, bins=50, alpha=0.85, edgecolor="black", color="steelblue")
    ax_weights.axvline(
        np.mean(weights),
        color="red",
        linestyle="--",
        lw=2,
        label=f"Mean={np.mean(weights):.2f}",
    )
    ax_weights.axvline(
        np.median(weights),
        color="orange",
        linestyle=":",
        lw=2,
        label=f"Median={np.median(weights):.2f}",
    )
    ax_weights.set_xlabel("Weight", fontsize=12)
    ax_weights.set_ylabel("Count", fontsize=12)
    ax_weights.set_title(
        f"Weight Distribution\nESS={int(info['ess'])}/{len(weights):,}, "
        f"clip={100*info['clip_fraction']:.1f}%",
        fontsize=11,
    )
    ax_weights.legend(fontsize=10)
    ax_weights.grid(True, alpha=0.3)

    plt.suptitle(
        "k-NN Halo Mass Function Weighting Test",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Plot] Saved: {save_path}")

    return fig


def _plot_environment_dist(
    ax: plt.Axes,
    parent_env: np.ndarray,
    subsample_env: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> None:
    """Plot environment distribution at a single scale.

    Args:
        ax: Matplotlib axes.
        parent_env: Parent (global) overdensities.
        subsample_env: Subsample overdensities.
        weights: Subsample weights.
        scale: Smoothing scale (Mpc/h).
    """
    bins = np.linspace(
        min(parent_env.min(), subsample_env.min()),
        max(parent_env.max(), subsample_env.max()),
        40,
    )

    ax.hist(
        parent_env,
        bins=bins,
        density=True,
        alpha=0.6,
        label="Parent (global)",
        histtype="stepfilled",
        color="gray",
    )
    ax.hist(
        subsample_env,
        bins=bins,
        density=True,
        alpha=0.7,
        label="Subsample (raw)",
        histtype="step",
        lw=2,
        color="red",
    )
    ax.hist(
        subsample_env,
        bins=bins,
        weights=weights,
        density=True,
        alpha=0.8,
        label="Subsample (weighted)",
        histtype="step",
        lw=2,
        color="blue",
    )

    ax.set_xlabel(f"log10(1 + δ) at R={scale:.2f} Mpc/h", fontsize=11)
    ax.set_ylabel("PDF", fontsize=11)
    ax.set_title(f"Environment (scale {scale:.2f})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def print_summary_statistics(
    truth_masses: np.ndarray,
    subsample_masses: np.ndarray,
    weights: np.ndarray,
    info: Dict,
) -> None:
    """Print summary statistics for the weighting test.

    Args:
        truth_masses: True halo masses (full sample).
        subsample_masses: Subsampled halo masses.
        weights: Subsample weights.
        info: Diagnostics from KNNReweighter.
    """
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nPopulation Sizes:")
    print(f"  Truth (full sample): {len(truth_masses):,} halos")
    print(f"  Subsample:           {len(subsample_masses):,} halos ({100*len(subsample_masses)/len(truth_masses):.1f}% of truth)")
    print(f"  ESS:                 {int(info['ess']):,} (effective sample size)")
    print(f"  ESS/N_subsample:     {info['ess']/len(subsample_masses):.1%}")

    print(f"\nk-NN Configuration:")
    print(f"  k:           {info['k']}")
    print(f"  Preprocess:  {info['preprocess']}")
    print(f"  Metric:      {info['metric']}")
    print(f"  Tau:         {info['tau']}")

    print(f"\nEnvironment Matching (KS distance):")
    print(f"  Mean across scales: {info['ks_mean']:.4f}")
    for i, (scale, ks) in enumerate(zip(info['scales'], info['ks_per_dim'])):
        print(f"    Scale {i} (R={scale:.2f}): {ks:.4f}")

    print(f"\nWeight Statistics:")
    print(f"  Mean:      {info['mean_weight']:.3f}")
    print(f"  Std:       {info['std_weight']:.3f}")
    print(f"  Min:       {info['min_weight']:.3f}")
    print(f"  Max:       {info['max_weight']:.3f}")
    print(f"  Clip frac: {100*info['clip_fraction']:.1f}%")
    print(f"  Clip range: {info['clip_range']}")

    # Mass statistics
    print(f"\nMass Range (log10 Msun):")
    print(f"  Truth:     {np.log10(truth_masses.min()):.2f} - {np.log10(truth_masses.max()):.2f}")
    print(f"  Subsample: {np.log10(subsample_masses.min()):.2f} - {np.log10(subsample_masses.max()):.2f}")

    # Weighted vs unweighted mean log mass
    mean_log_m_truth = np.mean(np.log10(truth_masses))
    mean_log_m_subsample = np.mean(np.log10(subsample_masses))
    mean_log_m_weighted = np.average(np.log10(subsample_masses), weights=weights)

    print(f"\nMean log10(M) [Msun]:")
    print(f"  Truth:      {mean_log_m_truth:.3f}")
    print(f"  Subsample:  {mean_log_m_subsample:.3f} (bias: {mean_log_m_subsample - mean_log_m_truth:+.3f})")
    print(f"  Weighted:   {mean_log_m_weighted:.3f} (bias: {mean_log_m_weighted - mean_log_m_truth:+.3f})")

    print("=" * 70 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test k-NN weighting on halo mass functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required inputs
    parser.add_argument(
        "--parent",
        type=str,
        required=True,
        help="Path to parent/global population HDF5 file (gridded overdensities, global reference)",
    )
    parser.add_argument(
        "--sample",
        type=str,
        required=True,
        help="Path to full sample HDF5 file (gridded overdensities, this is the truth)",
    )
    parser.add_argument(
        "--masses",
        type=str,
        required=True,
        help="Path to text file with full sample halo masses (one per line, truth HMF)",
    )

    # Subsampling
    parser.add_argument(
        "--subsample",
        type=int,
        required=True,
        help="Number of halos to randomly select from full sample for weighting test",
    )

    # k-NN parameters
    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of neighbors (default: 50)",
    )
    parser.add_argument(
        "--autotune_k",
        type=str,
        default="",
        help="Comma-separated k values for auto-tuning (e.g., '35,50,80,100')",
    )
    parser.add_argument(
        "--ess_floor",
        type=float,
        default=0.20,
        help="Minimum ESS fraction for auto-tuning (default: 0.20)",
    )
    parser.add_argument(
        "--preprocess",
        choices=["whiten", "standardize", "none"],
        default="whiten",
        help="Feature preprocessing mode (default: whiten)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature parameter for weights (default: 1.0)",
    )
    parser.add_argument(
        "--clip_min",
        type=float,
        default=0.01,
        help="Lower clip bound for weights (default: 0.01)",
    )
    parser.add_argument(
        "--clip_max",
        type=float,
        default=50.0,
        help="Upper clip bound for weights (default: 50.0)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save weighted masses as .npy file",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save diagnostic plot",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable interactive plotting",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for subsampling and diagnostics (default: 42)",
    )

    args = parser.parse_args()

    # Load overdensity features
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70 + "\n")

    parent_overdens, sample_overdens = load_multiscale_overdensities(
        parent_path=args.parent,
        sample_path=args.sample,
    )

    # Load halo masses (for the full sample = truth)
    truth_masses = load_halo_masses(args.masses)

    # Check consistency
    first_scale = next(iter(sample_overdens.values()))
    if len(first_scale) != len(truth_masses):
        raise ValueError(
            f"Mismatch: sample overdensities have {len(first_scale):,} entries "
            f"but mass file has {len(truth_masses):,} entries"
        )

    # Filter and transform parent data (removes delta <= -1, applies log transform)
    print("\n[Cleaning] Filtering and transforming parent overdensities...")
    parent_overdens, _ = filter_and_transform_overdensities(parent_overdens)

    # Additional NaN check on parent (shouldn't be any after filtering)
    parent_overdens = drop_nan_rows_per_scale(parent_overdens)

    # Filter and transform full sample data - need to synchronize with masses
    print("[Cleaning] Filtering and transforming full sample overdensities...")
    sample_overdens, mask = filter_and_transform_overdensities(sample_overdens)

    # Apply mask to masses to keep them synchronized
    truth_masses = truth_masses[mask]

    # Additional NaN check on sample (shouldn't be any after filtering)
    n_before = len(next(iter(sample_overdens.values())))
    sample_overdens = drop_nan_rows_per_scale(sample_overdens)
    n_after = len(next(iter(sample_overdens.values())))

    if n_after != n_before:
        # Need to rebuild mask for masses
        mask_nan = np.ones(n_before, dtype=bool)
        for arr in sample_overdens.values():
            # This shouldn't happen, but just in case
            pass
        print(f"[WARNING] Found {n_before - n_after} additional NaN values after transform!")

    # Subsample from the cleaned full sample
    if args.subsample is None:
        raise ValueError(
            "--subsample is required. Specify how many halos to randomly select "
            "from the full sample for weighting."
        )

    subsample_overdens, subsample_masses, subsample_indices = subsample_data(
        sample_overdens, truth_masses, args.subsample, args.random_state
    )

    # Run k-NN weighting
    # Goal: weight subsample to match parent (global) environment distribution
    print("\n" + "=" * 70)
    print("COMPUTING WEIGHTS")
    print("=" * 70 + "\n")

    auto_list = [int(x) for x in args.autotune_k.split(",") if x.strip().isdigit()]
    cfg = KNNReweighterConfig(
        k=args.k,
        autotune_k=auto_list if auto_list else None,
        ess_floor=args.ess_floor,
        preprocess=args.preprocess,
        tau=args.tau,
        clip_range=(args.clip_min, args.clip_max),
        random_state=args.random_state,
    )

    reweighter = KNNReweighter(cfg)
    weights, info = reweighter.fit(parent_overdens, subsample_overdens)

    # Print summary
    print_summary_statistics(truth_masses, subsample_masses, weights, info)

    # Save weighted masses if requested
    if args.output:
        weighted_data = np.column_stack([subsample_masses, weights])
        np.save(args.output, weighted_data)
        print(f"[Output] Saved (subsample_masses, weights) to: {args.output}")

    # Plot results
    if not args.no_plots:
        print("\n[Plot] Generating diagnostics...\n")
        fig = plot_hmf_comparison(
            truth_masses,
            subsample_masses,
            weights,
            info,
            parent_overdens,
            subsample_overdens,
            save_path=args.save,
        )
        plt.show()

    print("\n[Done] HMF weighting test complete.\n")


if __name__ == "__main__":
    main()
