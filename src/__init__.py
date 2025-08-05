"""
Efficient Attention Mechanisms package.
"""

__version__ = "0.1.0"

from .attention_implementations import StandardAttention, SparseAttention, FlashAttention
from .transformer_model import TransformerLM, TransformerBlock, PositionalEncoding
from .benchmark import PerformanceBenchmark, PerplexityEvaluator, run_comprehensive_benchmark

__all__ = [
    "StandardAttention",
    "SparseAttention", 
    "FlashAttention",
    "TransformerLM",
    "TransformerBlock",
    "PositionalEncoding",
    "PerformanceBenchmark",
    "PerplexityEvaluator",
    "run_comprehensive_benchmark"
]
