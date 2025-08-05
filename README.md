# Final project for the Development Tools for Scientific Computing exam. 
## Efficient Attention Mechanisms for Transformer Networks

A comprehensive implementation and comparison of attention mechanisms for transformer networks, focusing on performance optimization and high-performance computing (HPC) best practices.

## Project Overview

This project implements and compares three attention mechanisms:

1. **Standard Attention**: Traditional scaled dot-product attention ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762))
2. **Sparse Attention**: Attention with configurable sparsity patterns (local, strided, fixed)([Child et al., 2019](https://arxiv.org/abs/1904.10509))
3. **Flash Attention**: Memory-efficient attention using block-wise computation ([Dao et al., 2022](https://arxiv.org/abs/2205.14135))

### Mathematical Formulations

#### Standard Attention
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

#### Sparse Attention
```
Attention(Q, K, V) = softmax(mask ⊙ (QK^T / √d_k))V
```
where ⊙ denotes element-wise multiplication and mask defines the sparsity pattern.

#### Flash Attention
Uses block-wise computation to reduce memory complexity from O(n²) to O(n) while maintaining the same mathematical result as standard attention.

## Features

- **Multiple Attention Implementations**: Standard, sparse (local/strided/fixed patterns), and flash attention
- **Complete Transformer Architecture**: Full language model implementation with configurable attention
- **Comprehensive Benchmarking**: Runtime, memory usage, and computational complexity analysis
- **Profiling Tools**: GPU utilization, memory profiling, and bottleneck analysis
- **Extensive Testing**: Unit tests, integration tests, and numerical correctness verification
- **Performance Visualization**: Automated plotting and reporting

## Project Structure

```
├── experiments
│   └── config.yaml
├── pyproject.toml
├── README.md
├── report.md
├── results
│   ├── experiment_results
│   ├── profiling_results
│   └── sparse_attention_results
├── scripts
│   ├── profiler.py
│   ├── run.py
│   ├── sparse_attn_profiler.py
│   ├── train.py
│   └── verify.py
├── shell
│   └── submit.sh
├── src
│   ├── __init__.py
│   ├── __pycache__
│   ├── attention_implementations.py
│   ├── benchmark.py
│   └── transformer_model.py
├── tests
│   ├── __init__.py
│   ├── __pycache__
│   ├── conftest.py
│   ├── fixtures
│   ├── integration
│   ├── pytest.ini
│   ├── run_tests.py
│   └── unit
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone git@github.com:alexserra98/devtools_scicomp_project_2025.git
cd devtools_scicomp_project_2025
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

### Quick Start

Run a quick comparison of attention mechanisms:

```bash
python train.py --quick-test --output-dir quick_results
```

### Full Experiments

Run comprehensive experiments:

```bash
python train.py --output-dir experiment_results --max-epochs 10
```

### Benchmarking

Run performance benchmarks:

```bash
python -c "from src.benchmark import run_comprehensive_benchmark; run_comprehensive_benchmark()"
```

### Profiling

Run HPC profiling:

```bash
python profiler.py --output-dir profiling_results
```

### Testing

Run the test suite:

```bash
python -m pytest test/ -v
# or
python test/test_attention_suite.py
```

## Key Components

### Attention Implementations

#### StandardAttention
- Traditional multi-head attention
- O(n²) time and space complexity
- Baseline for comparison

#### SparseAttention
- Configurable sparsity patterns:
  - **Local**: Sliding window attention
  - **Strided**: Combination of local and strided patterns
  - **Fixed**: Custom sparsity masks
- Reduced computational complexity
- Configurable window sizes and stride patterns

#### FlashAttention
- Memory-efficient implementation
- Block-wise computation
- Same O(n²) FLOPs but O(n) memory
- Uses PyTorch's optimized implementation when available

### Transformer Model

Complete transformer implementation (`TransformerLM`) supporting:
- All attention mechanisms
- Configurable architecture (layers, heads, dimensions)
- Causal language modeling
- Text generation
- Training with PyTorch Lightning

### Benchmarking Suite

Comprehensive evaluation including:
- **Runtime Analysis**: Performance vs sequence length
- **Memory Profiling**: Peak and allocated memory usage
- **Perplexity Comparison**: Model quality evaluation
- **Scaling Analysis**: Computational complexity verification

### Profiling Tools

- **GPU Profiling**: CUDA kernel analysis
- **Memory Profiling**: Memory usage patterns
- **Computational Analysis**: FLOP counting and throughput
- **Bottleneck Identification**: Performance hotspots

## Results and Analysis

The benchmark suite generates several outputs:

1. **Performance Plots**: Runtime vs sequence length for all attention mechanisms
2. **Memory Usage Plots**: Memory consumption analysis
3. **Profiling Reports**: Detailed GPU and CPU profiling results
4. **Comparison Tables**: Side-by-side performance metrics


## Configuration

Key parameters can be configured via command line or configuration files:

### Model Parameters
- `d_model`: Model dimension (default: 512)
- `num_heads`: Number of attention heads (default: 8)
- `num_layers`: Number of transformer layers (default: 6)
- `max_seq_len`: Maximum sequence length (default: 1024)

### Training Parameters
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 5e-4)
- `max_epochs`: Maximum training epochs (default: 10)

### Attention-Specific Parameters
- `sparse_window_size`: Window size for sparse attention (default: 64)
- `sparse_stride`: Stride for strided attention (default: 8)
- `flash_block_size`: Block size for flash attention (default: 64)

## Experimental Results
I have conducted a series of experiments to test the 3 different attention implementation and full discussion is reported in `report.md`