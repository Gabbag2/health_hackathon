from .band_detection import (
    FREQ_BANDS,
    compute_band_envelope,
    detect_significant_band_epochs,
    apply_notch_filter,
    compute_broadband_power,
    compute_relative_band_power,
)
from .plot_bands import (
    BAND_COLORS,
    plot_band_activity,
)
from .preprocessing import (
    AMPLITUDE_THRESHOLD_UV,
    preprocess,
    load_raw,
    mark_amplitude_artifacts,
)
from .stats import (
    weighted_permutation_test,
    sig_label,
    eta_sq_label,
)

__all__ = [
    # Bandes de fréquence
    "FREQ_BANDS",
    # Visualisation
    "BAND_COLORS",
    "plot_band_activity",
    # Détection de bandes
    "compute_band_envelope",
    "detect_significant_band_epochs",
    # Filtrage et puissance
    "apply_notch_filter",
    "compute_broadband_power",
    "compute_relative_band_power",
    # Prétraitement
    "AMPLITUDE_THRESHOLD_UV",
    "preprocess",
    "load_raw",
    "mark_amplitude_artifacts",
    # Statistiques
    "weighted_permutation_test",
    "sig_label",
    "eta_sq_label",
]
