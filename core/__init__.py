from .band_detection import (
    FREQ_BANDS,
    compute_band_envelope,
    detect_significant_band_epochs,
)
from .plot_bands import (
    BAND_COLORS,
    plot_band_activity,
)

__all__ = [
    "FREQ_BANDS",
    "BAND_COLORS",
    "compute_band_envelope",
    "detect_significant_band_epochs",
    "plot_band_activity",
]
