#!/bin/bash
#SBATCH --job-name=comprehensive_benchmark
#SBATCH --partition=DGX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --gpus=1
#SBATCH -A lade
#SBATCH --output=/u/dssc/zenocosini/devtools_scicomp_project_2025/shell/.output/slurm-%j.out
#SBATCH --error=/u/dssc/zenocosini/devtools_scicomp_project_2025/shell/.output/slurm-%j.err

# Set environment variables to suppress warnings
export TOKENIZERS_PARALLELISM=false
export WANDB_SILENT=true

echo " Starting Comprehensive Training and Benchmarking Pipeline"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "Time: $(date)"
echo "Working directory: $(pwd)"

cd /u/dssc/zenocosini/devtools_scicomp_project_2025
source /u/dssc/zenocosini/devtools_scicomp_project_2025/venv/bin/activate

echo ""
echo " Environment Information:"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "CUDA Device: $(python -c 'import torch; print(torch.cuda.get_device_name())')"
    echo "CUDA Memory: $(python -c 'import torch; print(f\"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\")')"
fi

echo ""
echo " PHASE 1: Comprehensive Benchmark Suite"
echo "=========================================="
echo "Running unified benchmark with extended sequence lengths..."

# Run the comprehensive benchmark
python src/benchmark.py \
    --output-dir experiment_results/comprehensive_benchmark \
    --device auto

echo ""
echo "🏋️ PHASE 2: Model Training and Comparison"
echo "=========================================="
echo "Training models with different attention mechanisms..."

# Run training with different attention mechanisms
python scripts/train.py \
    --output-dir experiment_results/training \
    --max-epochs 10

echo ""
echo " PHASE 3: Additional Profiling and Analysis"
echo "============================================="
echo "Running detailed profiling..."

# Run profiler for additional insights
python scripts/profiler.py \
    --output-dir experiment_results/profiling

echo ""
echo "✅ PIPELINE COMPLETED SUCCESSFULLY"
echo "=================================="
echo "Results saved to:"
echo "  - Comprehensive Benchmark: experiment_results/comprehensive_benchmark/"
echo "  - Training Results: experiment_results/training/"
echo "  - Profiling Results: experiment_results/profiling/"
echo ""
echo "End time: $(date)"
