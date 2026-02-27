"""
plot_bands.py
-------------
Visualise where theta, alpha, and beta activity is significantly present
inside an LFP epoch.

Main entry point
----------------
    plot_band_activity(epoch_row, ...)

Layout
------
  Row 0  — Raw LFP + translucent coloured overlays for each significant band.
  Row 1+ — One sub-panel per band: smoothed amplitude envelope with a dashed
            threshold line.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .band_detection import detect_significant_band_epochs, FREQ_BANDS


# ---------------------------------------------------------------------------
# Colour scheme (one colour per band)
# ---------------------------------------------------------------------------

BAND_COLORS = {
    "theta": "#FF9500",   # orange
    "alpha": "#30D158",   # green
    "beta":  "#0A84FF",   # blue
}


# ---------------------------------------------------------------------------
# Public plotting function
# ---------------------------------------------------------------------------

def plot_band_activity(
    epoch_row,
    channel="dHPC_lfp",
    fs=1250,
    smooth_sec=0.1,
    threshold_percentile=75,
    xlim=None,
    bands=None,
    title_prefix="",
):
    """
    Plot an LFP signal with coloured overlays and per-band envelope panels
    that highlight windows of significant alpha, theta, and beta activity.

    Parameters
    ----------
    epoch_row : pd.Series
        One row from the LFP DataFrame (as returned by `select_epoch`).
        Must contain the columns `t_start`, `t_end`, and `channel`.
    channel : str
        Column name of the LFP channel to analyse.
        One of 'dHPC_lfp', 'vHPC_lfp', 'bla_lfp'.
    fs : float
        Sampling frequency (Hz).  Default 1250.
    smooth_sec : float
        Duration (s) of the smoothing window applied to the Hilbert envelope
        before thresholding.  Larger values → smoother, less local detection.
    threshold_percentile : float
        Percentile (0–100) used to binarise each band's envelope.
        E.g. 75 flags the top 25 % of envelope values as significant.
    xlim : tuple (t_start, t_end) or None
        Zoom window in seconds.  None → full epoch.
    bands : dict or None
        Override default frequency band definitions.  Keys are band names,
        values are (f_low, f_high) tuples in Hz.  Defaults to FREQ_BANDS.
    title_prefix : str
        Optional label prepended to the figure title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, so callers can save or further customise it.

    Examples
    --------
    >>> fig = plot_band_activity(
    ...     epoch_row=epoch,
    ...     channel="dHPC_lfp",
    ...     xlim=(epoch["t_start"] + 10, epoch["t_start"] + 15),
    ...     threshold_percentile=75,
    ...     title_prefix="NREM I — session 08",
    ... )
    >>> plt.show()
    """
    if bands is None:
        bands = FREQ_BANDS

    lfp = np.asarray(epoch_row[channel])
    t_start = float(epoch_row["t_start"])
    t_end   = float(epoch_row["t_end"])

    # Run detection
    result = detect_significant_band_epochs(
        lfp, fs,
        t_start=t_start,
        smooth_sec=smooth_sec,
        threshold_percentile=threshold_percentile,
        bands=bands,
    )

    times      = result["times"]
    envelopes  = result["envelopes"]
    significant = result["significant"]
    thresholds = result["thresholds"]
    band_names = list(bands.keys())
    n_bands    = len(band_names)

    # -----------------------------------------------------------------------
    # Figure layout: 1 LFP panel on top + 1 envelope panel per band below
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=1 + n_bands,
        ncols=1,
        figsize=(18, 3 + 2.2 * (1 + n_bands)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5] + [1] * n_bands},
    )

    # -----------------------------------------------------------------------
    # Top panel: raw LFP + coloured band overlays
    # -----------------------------------------------------------------------
    ax_lfp = axes[0]
    ax_lfp.plot(times, lfp, lw=0.6, color="steelblue", zorder=2)
    ax_lfp.set_ylabel(f"{channel}\n(µV)", fontsize=9)

    title = (f"{title_prefix} — " if title_prefix else "") + (
        f"Significant band activity  "
        f"(Hilbert envelope ≥ {threshold_percentile}th percentile)"
    )
    ax_lfp.set_title(title, fontsize=10)

    # Draw a semi-transparent span for every significant sample, per band.
    # We merge consecutive True samples into contiguous intervals to keep the
    # number of axvspan calls small (faster rendering on large epochs).
    dt = times[1] - times[0]  # sample period

    for name in band_names:
        color = BAND_COLORS.get(name, "grey")
        sig   = significant[name]

        # Find contiguous True runs
        changes = np.diff(sig.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1

        # Handle edges
        if sig[0]:
            starts = np.concatenate([[0], starts])
        if sig[-1]:
            ends = np.concatenate([ends, [len(sig)]])

        for s, e in zip(starts, ends):
            ax_lfp.axvspan(times[s], times[e - 1] + dt,
                           alpha=0.18, color=color, lw=0, zorder=1)

    # Legend
    patches = [
        mpatches.Patch(
            color=BAND_COLORS.get(n, "grey"), alpha=0.5,
            label=f"{n.capitalize()}  ({bands[n][0]}–{bands[n][1]} Hz)"
        )
        for n in band_names
    ]
    ax_lfp.legend(handles=patches, loc="upper right", fontsize=8)

    # -----------------------------------------------------------------------
    # Lower panels: per-band envelope + threshold
    # -----------------------------------------------------------------------
    for i, name in enumerate(band_names):
        ax    = axes[1 + i]
        color = BAND_COLORS.get(name, "grey")
        env   = envelopes[name]
        thr   = thresholds[name]

        ax.fill_between(times, env, alpha=0.55, color=color, lw=0)
        ax.plot(times, env, lw=0.5, color=color)
        ax.axhline(
            thr, color="black", lw=1.2, ls="--",
            label=f"{threshold_percentile}th pct  ({thr:.1f} µV)"
        )
        ax.set_ylabel(f"{name.capitalize()}\nenvelope (µV)", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    if xlim is not None:
        for ax in axes:
            ax.set_xlim(xlim)

    plt.tight_layout()
    return fig
