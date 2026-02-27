"""
band_detection.py
-----------------
Detect time periods in an LFP epoch where a specific EEG frequency band
(theta, alpha, or beta) is significantly active.

Approach
--------
For each band we:
  1. Design a Butterworth bandpass filter (same strategy as the notebook).
  2. Apply the filter with filtfilt (zero-phase, no delay).
  3. Compute the instantaneous amplitude envelope via the Hilbert transform.
  4. Smooth the envelope with a short sliding window (reduces noise).
  5. Threshold: a sample is "significant" when its smoothed envelope exceeds
     the requested percentile of the full-epoch envelope.

Frequency bands (defaults, all in Hz)
--------------------------------------
  theta : 4–8 Hz   (rodent theta can extend to ~12 Hz; adjust via `bands`)
  alpha : 8–13 Hz
  beta  : 13–30 Hz
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


# ---------------------------------------------------------------------------
# Default band definitions
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "theta": (4, 8),    # Hz
    "alpha": (8, 13),   # Hz
    "beta":  (13, 30),  # Hz
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_band_envelope(lfp, fs, bands=None, smooth_sec=0.1):
    """
    Compute the instantaneous amplitude envelope for each frequency band.

    Parameters
    ----------
    lfp : np.ndarray, shape (n_samples,)
        Raw LFP signal (µV).
    fs : float
        Sampling frequency (Hz).
    bands : dict or None
        Frequency bands as {name: (f_low, f_high)}.  Defaults to FREQ_BANDS.
    smooth_sec : float
        Duration (s) of the rectangular smoothing kernel applied to the
        envelope.  Set to 0 to disable smoothing.

    Returns
    -------
    envelopes : dict
        {band_name: np.ndarray of shape (n_samples,)} — smoothed amplitude
        envelope for each band.
    """
    if bands is None:
        bands = FREQ_BANDS

    nyquist = fs / 2.0
    envelopes = {}

    for name, (f_low, f_high) in bands.items():
        # Clamp to valid range for the Butterworth design
        lo = max(f_low,  0.5) / nyquist
        hi = min(f_high, nyquist - 0.5) / nyquist

        b, a = butter(4, [lo, hi], btype="bandpass")
        filtered = filtfilt(b, a, lfp)

        # Instantaneous amplitude from analytic signal
        envelope = np.abs(hilbert(filtered))

        # Optional smoothing
        if smooth_sec > 0:
            n_smooth = max(1, int(smooth_sec * fs))
            kernel = np.ones(n_smooth) / n_smooth
            envelope = np.convolve(envelope, kernel, mode="same")

        envelopes[name] = envelope

    return envelopes


def detect_significant_band_epochs(
    lfp,
    fs,
    t_start=0.0,
    smooth_sec=0.1,
    threshold_percentile=75,
    bands=None,
):
    """
    Detect samples where each frequency band is significantly active.

    A sample is considered significant for a given band when its smoothed
    Hilbert envelope exceeds the `threshold_percentile`-th percentile of
    the full-epoch envelope for that band.

    Parameters
    ----------
    lfp : np.ndarray
        Raw LFP signal.
    fs : float
        Sampling frequency (Hz).
    t_start : float
        Absolute time of the first sample (s).  Used to build the time axis.
    smooth_sec : float
        Smoothing window for the envelope (s).
    threshold_percentile : float
        Percentile threshold (0–100).  75 means "top 25 % of the epoch".
    bands : dict or None
        Frequency band definitions.  Defaults to FREQ_BANDS.

    Returns
    -------
    result : dict with keys
        "times"       : np.ndarray, shape (n_samples,) — absolute time axis.
        "envelopes"   : dict {band: np.ndarray} — smoothed amplitude envelope.
        "significant" : dict {band: np.ndarray bool} — True where significant.
        "thresholds"  : dict {band: float} — amplitude threshold per band.
    """
    if bands is None:
        bands = FREQ_BANDS

    envelopes = compute_band_envelope(lfp, fs, bands=bands, smooth_sec=smooth_sec)

    times = np.linspace(t_start, t_start + len(lfp) / fs, len(lfp), endpoint=False)

    significant = {}
    thresholds = {}

    for name, env in envelopes.items():
        thr = np.percentile(env, threshold_percentile)
        thresholds[name] = thr
        significant[name] = env >= thr

    return {
        "times":       times,
        "envelopes":   envelopes,
        "significant": significant,
        "thresholds":  thresholds,
    }
