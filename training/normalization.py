"""Public normalization API for the BC/SAC training pipeline.

The implementation lives beside the HDF5 loader so that the exact same
statistics object is used by inspection, BC pretraining, and future SAC.
"""

try:
    from .dataset import NormalizationStats
except ImportError:  # direct execution with training/ on sys.path
    from dataset import NormalizationStats

__all__ = ["NormalizationStats"]
