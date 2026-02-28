"""
preprocessing.py
----------------
Chargement et prétraitement des enregistrements bruts MNE pour ce projet.

Pipeline appliqué à chaque fichier chargé
------------------------------------------
1. Référence bipolaire EMG  : EMG 1 − EMG 2  → canal "EMG" (type emg)
2. Référence EEG            : re-référencement sur A2, puis suppression de A2
3. Sauvegarde de la fréquence d'échantillonnage originale
4. Rééchantillonnage        : 128 Hz
5. Filtre coupe-bande        : 50 Hz (secteur) et harmoniques jusqu'à Nyquist
6. Filtre passe-bande        : 0.3 – 35 Hz
7. Rejet d'artifacts         : epochs dont l'amplitude EEG dépasse 500 µV
                               sont marquées (colonne "bad_amplitude")

Usage
-----
    import mne
    from core import preprocess, load_raw

    raw, sfreq_orig = preprocess(mne.io.read_raw_edf("recording.edf", preload=True))

    # Ou directement :
    raw, sfreq_orig = load_raw("recording.edf")
"""

import mne
import numpy as np

# Seuil d'amplitude au-delà duquel une epoch EEG est considérée artefactée (µV)
AMPLITUDE_THRESHOLD_UV = 500.0


def preprocess(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, float]:
    """
    Applique le pipeline de prétraitement standard sur un objet MNE Raw.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Enregistrement brut MNE chargé avec preload=True.
        Doit contenir les canaux : "EMG 1", "EMG 2", "A2".

    Returns
    -------
    raw : mne.io.BaseRaw
        Enregistrement prétraité (modifié en place).
    sfreq_orig : float
        Fréquence d'échantillonnage originale avant rééchantillonnage (Hz).
    """
    # 1. Référence bipolaire EMG
    mne.set_bipolar_reference(raw, "EMG 1", "EMG 2", ch_name="EMG", copy=False)
    raw.set_channel_types({"EMG": "emg"})

    # 2. Référence EEG sur A2, puis supprimer A2
    mne.set_eeg_reference(raw, ["A2"], copy=False)
    raw.drop_channels(["A2"])

    # 3. Sauvegarder la fréquence originale
    sfreq_orig = raw.info["sfreq"]

    # 4. Rééchantillonnage à 128 Hz
    raw.resample(128, npad="auto")

    # 5. Filtre coupe-bande 50 Hz + harmoniques (60, 100, 150 Hz si < Nyquist)
    nyquist = raw.info["sfreq"] / 2.0
    notch_freqs = [f for f in [50, 100, 150] if f < nyquist]
    raw.notch_filter(notch_freqs)

    # 6. Filtre passe-bande 0.3 – 35 Hz
    raw.filter(l_freq=0.3, h_freq=35.0, n_jobs=-1)

    return raw, sfreq_orig


def load_raw(filepath: str, **kwargs) -> tuple[mne.io.BaseRaw, float]:
    """
    Charge un fichier EEG brut et applique immédiatement le pipeline preprocess().

    Formats supportés : .edf, .fif, .bdf, .gdf (auto-détecté par MNE).

    Parameters
    ----------
    filepath : str
        Chemin vers le fichier brut.
    **kwargs
        Arguments supplémentaires passés à la fonction de lecture MNE
        (ex. : stim_channel=False, exclude=[...]).

    Returns
    -------
    raw : mne.io.BaseRaw
        Enregistrement prétraité.
    sfreq_orig : float
        Fréquence d'échantillonnage originale (avant rééchantillonnage).

    Example
    -------
    >>> raw, sfreq_orig = load_raw("data/session_01.edf")
    >>> print(f"Fréquence originale : {sfreq_orig} Hz")
    >>> print(raw.info)
    """
    raw = mne.io.read_raw(filepath, preload=True, **kwargs)
    return preprocess(raw)


def mark_amplitude_artifacts(
    epochs: mne.Epochs,
    threshold_uv: float = AMPLITUDE_THRESHOLD_UV,
    picks: str = "eeg",
) -> np.ndarray:
    """
    Identifie les epochs dont l'amplitude EEG dépasse le seuil.

    Parameters
    ----------
    epochs : mne.Epochs
    threshold_uv : float
        Seuil en µV (défaut : 500 µV).
    picks : str
        Sélection de canaux passée à mne.pick_types (défaut : "eeg").

    Returns
    -------
    bad_mask : np.ndarray of bool, shape (n_epochs,)
        True si l'epoch dépasse le seuil sur au moins un canal EEG.
    """
    data_uv = epochs.get_data(picks=picks) * 1e6  # Volts → µV
    peak_to_peak = data_uv.max(axis=-1) - data_uv.min(axis=-1)  # (n_epochs, n_channels)
    bad_mask = (peak_to_peak > threshold_uv).any(axis=1)
    return bad_mask
