"""KeyRecon: author-keyword reconstruction with Adaptive CKS."""

from .version import __version__
from .engine import KeyRecon
from .config import AdaptiveCKSWeights, EN_REFERENCE_WEIGHTS

__all__ = [
    "__version__",
    "KeyRecon",
    "AdaptiveCKSWeights",
    "EN_REFERENCE_WEIGHTS",
    "KoreanReference",
    "load_korean_reference",
    "KoreanAdaptiveCKS",
    "KoreanRunResult",
    "run_korean_adaptive",
]

from .ko_reference import KoreanReference, load_korean_reference
from .ko_runtime import KoreanAdaptiveCKS, KoreanRunResult, run_korean_adaptive
