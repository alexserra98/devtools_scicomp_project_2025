"""
Test suite for the devtools_scicomp_project_2025 package.

This test suite provides comprehensive coverage of:
- Attention mechanism implementations
- Transformer model components
- Benchmark utilities
- Performance validation
"""

import os
import sys

# Add src to Python path for testing
test_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(test_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
