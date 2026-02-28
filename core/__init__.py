from .band_detection import (
    FREQ_BANDS,
    compute_band_envelope,
    detect_significant_band_epochs,
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

__all__ = [
    # Bandes de fréquence
    "FREQ_BANDS",
    # Visualisation
    "BAND_COLORS",
    "plot_band_activity",
    # Détection de bandes
    "compute_band_envelope",
    "detect_significant_band_epochs",
    # Prétraitement
    "AMPLITUDE_THRESHOLD_UV",
    "preprocess",
    "load_raw",
    "mark_amplitude_artifacts",
]
