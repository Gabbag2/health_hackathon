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
  low   : 0.2-7 Hz
  theta : 7-12 Hz
  beta  : 15-30 Hz
  High  : 30-80 Hz
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, iirnotch, welch


# ---------------------------------------------------------------------------
# Default band definitions
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    "low":   (0.2, 7),  # Hz
    "theta": (7, 12),   # Hz
    "beta":  (15, 30),  # Hz
    "high":  (30, 80),  # Hz
#    "ripple150": (130, 160),  # Hz
#    "ripple250": (230, 270),  # Hz
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


# ---------------------------------------------------------------------------
# Filtre notch (coupe-bande)
# ---------------------------------------------------------------------------

def apply_notch_filter(lfp, fs, notch_freq=50.0, Q=30):
    """
    Applique un filtre coupe-bande (notch) à notch_freq Hz.

    Utilise scipy.signal.iirnotch + filtfilt (zero-phase).
    Le facteur de qualité Q contrôle la largeur de bande :
    bande_3dB = notch_freq / Q  (≈ 1.67 Hz à Q=30, f=50 Hz).

    Paramètres
    ----------
    lfp : np.ndarray, shape (n_samples,)
        Signal LFP brut (µV), typiquement à 1250 Hz.
    fs : float
        Fréquence d'échantillonnage (Hz).
    notch_freq : float
        Fréquence cible du notch (Hz). Défaut : 50 Hz (bruit secteur).
    Q : float
        Facteur de qualité. Défaut : 30.

    Retourne
    --------
    lfp_filt : np.ndarray, shape (n_samples,)
        Signal après suppression de la composante à notch_freq Hz.
    """
    lfp = np.asarray(lfp, dtype=float)
    b, a = iirnotch(notch_freq, Q, fs)
    # filtfilt requiert len(x) > padlen = 3 * max(len(a), len(b))
    padlen = 3 * max(len(a), len(b))
    if len(lfp) <= padlen:
        return lfp.copy()
    return filtfilt(b, a, lfp)


# ---------------------------------------------------------------------------
# Puissance broadband (Welch)
# ---------------------------------------------------------------------------

def compute_broadband_power(lfp, fs, f_low=0.5, f_high=50.0, nperseg=None):
    """
    Calcule la puissance totale PSD Welch dans la bande [f_low, f_high].

    La puissance est exprimée en µV² (somme des densités spectrales × Δf).

    Paramètres
    ----------
    lfp : np.ndarray
        Signal LFP (typiquement après filtre notch 50 Hz).
    fs : float
        Fréquence d'échantillonnage (Hz).
    f_low : float
        Borne basse de la bande broadband (Hz). Défaut : 0.5 Hz.
    f_high : float
        Borne haute (Hz). Défaut : 50 Hz.
    nperseg : int or None
        Longueur du segment Welch. Si None, utilise min(4*fs, n_samples).
        Clampé à n_samples pour éviter les UserWarnings scipy.

    Retourne
    --------
    power : float
        Puissance broadband en µV². np.nan si le signal est trop court
        ou si la puissance calculée n'est pas finie.
    """
    lfp = np.asarray(lfp, dtype=float)
    n = len(lfp)
    if n < 2:
        return np.nan
    _nperseg = min(nperseg if nperseg is not None else int(4 * fs), n)
    _noverlap = min(_nperseg // 2, _nperseg - 1)
    freqs, psd = welch(lfp, fs=fs, nperseg=_nperseg, noverlap=_noverlap)
    total = psd[(freqs >= f_low) & (freqs <= f_high)].sum()
    return float(total) if np.isfinite(total) else np.nan


# ---------------------------------------------------------------------------
# Puissance relative par bande (Welch)
# ---------------------------------------------------------------------------

def compute_relative_band_power(
    lfp, fs, f_low, f_high,
    f_total_low=0.5, f_total_high=50.0,
    nperseg=None,
):
    """
    Calcule la puissance relative d'une bande via la méthode de Welch.

    Puissance relative = Σ PSD(f∈[f_low, f_high]) / Σ PSD(f∈[f_total_low, f_total_high])

    Paramètres
    ----------
    lfp : np.ndarray
        Signal LFP (typiquement après filtre notch 50 Hz).
    fs : float
        Fréquence d'échantillonnage (Hz).
    f_low, f_high : float
        Bornes de la bande d'intérêt (Hz).
    f_total_low, f_total_high : float
        Bornes pour le calcul de la puissance totale de référence.
        Défaut : 0.5–50 Hz (cohérent avec le filtre notch appliqué).
    nperseg : int or None
        Longueur du segment Welch. Si None, utilise min(4*fs, n_samples).

    Retourne
    --------
    rel_power : float
        Puissance relative (adimensionnelle, entre 0 et 1).
        np.nan si le signal est trop court ou la puissance totale est nulle.
    """
    lfp = np.asarray(lfp, dtype=float)
    n = len(lfp)
    if n < 2:
        return np.nan
    _nperseg = min(nperseg if nperseg is not None else int(4 * fs), n)
    _noverlap = min(_nperseg // 2, _nperseg - 1)
    freqs, psd = welch(lfp, fs=fs, nperseg=_nperseg, noverlap=_noverlap)
    total = psd[(freqs >= f_total_low) & (freqs <= f_total_high)].sum()
    if total == 0 or not np.isfinite(total):
        return np.nan
    band_power = psd[(freqs >= f_low) & (freqs <= f_high)].sum()
    return float(band_power / total)
