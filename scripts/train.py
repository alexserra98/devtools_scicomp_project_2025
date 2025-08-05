import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import wandb
from pytorch_lightning.loggers import WandbLogger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.transformer_model import TransformerLM, count_parameters
from src.benchmark import (
    TextDataset,
    SyntheticTextDataset,
    PerformanceBenchmark,
    PerplexityEvaluator,
    run_comprehensive_benchmark,
)
from src.attention_implementations import (
    StandardAttention,
    SparseAttention,
    FlashAttention,
)


class TrainingConfig:
    """Configuration for training experiments."""

    def __init__(self):
        # Tokenizer settings
        self.tokenizer_name = "gpt2"  # Use GPT-2 tokenizer
        self.use_synthetic = False  # Flag to use synthetic vs real text

        # Model parameters (vocab_size will be set from tokenizer)
        self.vocab_size = 50257  # GPT-2 vocab size, will be updated from tokenizer
        self.d_model = 512
        self.num_heads = 8
        self.num_layers = 6
        self.d_ff = 2048
        self.max_seq_len = 1024
        self.dropout = 0.1

        # Training parameters
        self.batch_size = 8
        self.learning_rate = 5e-4
        self.warmup_steps = 2000
        self.max_epochs = 10
        self.gradient_clip_val = 1.0

        # Data parameters
        self.train_samples = 5000
        self.val_samples = 1000
        self.seq_len = (
            1024  # Increased for better demonstration of efficiency differences
        )

        # Attention-specific parameters
        self.sparse_window_size = 128  # Increased for more realistic local attention
        self.sparse_stride = 16  # Reduced stride for better performance
        self.flash_block_size = 128  # Increased block size for better efficiency


class ExperimentRunner:
    """Run comprehensive experiments comparing attention mechanisms."""

    def __init__(
        self,
        config: TrainingConfig,
        output_dir: str,
        device: str = "auto",
        use_wandb: bool = False,
        wandb_project: str = "attention-comparison",
        skip_long_context: bool = False,
        max_benchmark_seq_len: int = 4096,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.skip_long_context = skip_long_context
        self.max_benchmark_seq_len = max_benchmark_seq_len

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                config={
                    "vocab_size": config.vocab_size,
                    "d_model": config.d_model,
                    "num_heads": config.num_heads,
                    "num_layers": config.num_layers,
                    "d_ff": config.d_ff,
                    "max_seq_len": config.max_seq_len,
                    "dropout": config.dropout,
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "max_epochs": config.max_epochs,
                    "train_samples": config.train_samples,
                    "val_samples": config.val_samples,
                    "seq_len": config.seq_len,
                    "use_synthetic": config.use_synthetic,
                    "tokenizer_name": config.tokenizer_name
                    if not config.use_synthetic
                    else None,
                    "sparse_window_size": config.sparse_window_size,
                    "sparse_stride": config.sparse_stride,
                    "flash_block_size": config.flash_block_size,
                },
                name=f"attention-comparison-{time.strftime('%Y%m%d-%H%M%S')}",
                tags=["attention", "transformer", "comparison"],
            )

        print(f"Using device: {self.device}")

        # Model configurations
        self.model_configs = {
            "standard": {
                "attention_type": "standard",
                "description": "Standard multi-head attention",
            },
            "sparse_local": {
                "attention_type": "sparse",
                "sparsity_pattern": "local",
                "window_size": config.sparse_window_size,
                "description": f"Sparse attention with local window (size={config.sparse_window_size})",
            },
            "sparse_strided": {
                "attention_type": "sparse",
                "sparsity_pattern": "strided",
                "window_size": config.sparse_window_size,
                "stride": config.sparse_stride,
                "description": f"Sparse attention with strided pattern (window={config.sparse_window_size}, stride={config.sparse_stride})",
            },
            "flash": {
                "attention_type": "flash",
                "block_size": config.flash_block_size,
                "description": f"Flash attention (block_size={config.flash_block_size})",
            },
        }

        self.results = {}

        # Initialize tokenizer if using real text
        if not self.config.use_synthetic:
            print(f"Loading tokenizer: {self.config.tokenizer_name}")
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Update vocab size in config
            self.config.vocab_size = self.tokenizer.vocab_size
            print(f"Tokenizer vocabulary size: {self.config.vocab_size}")
        else:
            self.tokenizer = None

    def create_datasets(self):
        """Create training and validation datasets."""
        if not self.config.use_synthetic and self.tokenizer is not None:
            print("Creating real text datasets...")

            train_dataset = TextDataset(
                self.tokenizer,
                self.config.seq_len,
                self.config.train_samples,
                dataset_name="wikitext",
                split="train",
            )

            val_dataset = TextDataset(
                self.tokenizer,
                self.config.seq_len,
                self.config.val_samples,
                dataset_name="wikitext",
                split="validation",
            )
        else:
            print("Creating synthetic datasets...")

            train_dataset = SyntheticTextDataset(
                self.config.vocab_size, self.config.seq_len, self.config.train_samples
            )

            val_dataset = SyntheticTextDataset(
                self.config.vocab_size, self.config.seq_len, self.config.val_samples
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        return train_loader, val_loader

    def train_model(
        self, model_name: str, model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Train a single model and return metrics."""
        print(f"\\nTraining {model_name} model...")
        print(f"Configuration: {model_config['description']}")

        # Create model
        attention_kwargs = {
            k: v
            for k, v in model_config.items()
            if k not in ["attention_type", "description"]
        }

        model = TransformerLM(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            attention_type=model_config["attention_type"],
            dropout=self.config.dropout,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            **attention_kwargs,
        )

        # Count parameters
        num_params = count_parameters(model)
        print(f"Model parameters: {num_params:,}")

        # Create datasets
        train_loader, val_loader = self.create_datasets()

        # Setup trainer
        logger = None
        if self.use_wandb:
            logger = WandbLogger(
                project=self.wandb_project,
                name=f"{model_name}-{time.strftime('%Y%m%d-%H%M%S')}",
                tags=["attention", "transformer", model_name],
            )
            # Log model configuration to wandb
            wandb.log(
                {
                    f"{model_name}_config": model_config,
                    f"{model_name}_num_parameters": num_params,
                }
            )

        trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            gradient_clip_val=self.config.gradient_clip_val,
            enable_progress_bar=True,
            enable_model_summary=False,
            default_root_dir=self.output_dir / model_name,
            logger=logger
            if self.use_wandb
            else pl.loggers.CSVLogger(self.output_dir / model_name, name="logs"),
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1,
                    filename="best-{epoch:02d}-{val_loss:.3f}",
                ),
                pl.callbacks.EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ],
        )

        # Measure training time
        start_time = time.time()
        trainer.fit(model, train_loader, val_loader)
        training_time = time.time() - start_time

        # Get final metrics
        final_train_loss = trainer.callback_metrics.get("train_loss", float("inf"))
        final_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        final_train_perplexity = trainer.callback_metrics.get(
            "train_perplexity", float("inf")
        )
        final_val_perplexity = trainer.callback_metrics.get(
            "val_perplexity", float("inf")
        )

        # Benchmark inference
        model.eval()
        model = model.to(self.device)

        with torch.no_grad():
            # Create test batch
            if not self.config.use_synthetic and self.tokenizer is not None:
                # Create test batch with real tokenized text
                test_text = "This is a test sentence for inference benchmarking. " * (
                    self.config.seq_len // 10
                )
                test_tokens = self.tokenizer.encode(
                    test_text,
                    max_length=self.config.seq_len,
                    truncation=True,
                    padding="max_length",
                )
                test_input = torch.tensor(
                    [test_tokens] * self.config.batch_size,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                # Use random tokens for synthetic data
                test_input = torch.randint(
                    0,
                    self.config.vocab_size,
                    (self.config.batch_size, self.config.seq_len),
                    device=self.device,
                )

            # Warmup
            for _ in range(10):
                _ = model(test_input)

            # Measure inference time
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            for _ in range(100):
                _ = model(test_input)

            if self.device.type == "cuda":
                torch.cuda.synchronize()

            inference_time = (time.time() - start_time) / 100 * 1000  # ms per inference

        # Log metrics to wandb if enabled
        if self.use_wandb:
            wandb.log(
                {
                    f"{model_name}_num_parameters": num_params,
                    f"{model_name}_training_time_seconds": training_time,
                    f"{model_name}_inference_time_ms": inference_time,
                    f"{model_name}_final_train_loss": float(final_train_loss)
                    if torch.is_tensor(final_train_loss)
                    else final_train_loss,
                    f"{model_name}_final_val_loss": float(final_val_loss)
                    if torch.is_tensor(final_val_loss)
                    else final_val_loss,
                    f"{model_name}_final_train_perplexity": float(
                        final_train_perplexity
                    )
                    if torch.is_tensor(final_train_perplexity)
                    else final_train_perplexity,
                    f"{model_name}_final_val_perplexity": float(final_val_perplexity)
                    if torch.is_tensor(final_val_perplexity)
                    else final_val_perplexity,
                }
            )

        return {
            "num_parameters": num_params,
            "training_time_seconds": training_time,
            "inference_time_ms": inference_time,
            "final_train_loss": float(final_train_loss)
            if torch.is_tensor(final_train_loss)
            else final_train_loss,
            "final_val_loss": float(final_val_loss)
            if torch.is_tensor(final_val_loss)
            else final_val_loss,
            "final_train_perplexity": float(final_train_perplexity)
            if torch.is_tensor(final_train_perplexity)
            else final_train_perplexity,
            "final_val_perplexity": float(final_val_perplexity)
            if torch.is_tensor(final_val_perplexity)
            else final_val_perplexity,
            "config": model_config,
        }

    def run_experiments(self):
        """Run all experiments."""
        print("=" * 60)
        print("STARTING ATTENTION MECHANISM COMPARISON EXPERIMENTS")
        print("=" * 60)

        # Train all models
        for model_name, model_config in self.model_configs.items():
            try:
                metrics = self.train_model(model_name, model_config)
                self.results[model_name] = metrics

                # Clear cache between models
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                self.results[model_name] = {"error": str(e)}

        # Save results
        self.save_results()

        # Log final comparison metrics to wandb
        if self.use_wandb:
            self.log_comparison_metrics()

        # Run performance benchmarks
        print("\\n" + "=" * 60)
        print("RUNNING PERFORMANCE BENCHMARKS")
        print("=" * 60)

        try:
            run_comprehensive_benchmark(
                output_dir=str(self.output_dir / "benchmarks"), device=str(self.device)
            )
        except Exception as e:
            print(f"Error running benchmarks: {str(e)}")

        # Run long-context benchmark to show where sparse attention excels
        if not self.skip_long_context:
            print("\\n" + "=" * 60)
            print("RUNNING LONG-CONTEXT BENCHMARK")
            print("=" * 60)
            print(
                "This benchmark demonstrates scenarios where sparse attention outperforms standard attention"
            )

            try:
                from scripts.long_context_benchmark import LongContextBenchmark

                long_benchmark = LongContextBenchmark(str(self.device))

                # Run a focused test showing sparse attention advantages
                max_seq_len = min(
                    self.max_benchmark_seq_len,
                    4096 if torch.cuda.is_available() else 1024,
                )
                long_benchmark.benchmark_memory_scaling(max_seq_len)
                long_benchmark.benchmark_sparsity_patterns(max_seq_len)

                # Save results
                long_benchmark.save_results(self.output_dir / "long_context")
                long_benchmark.plot_results(self.output_dir / "long_context")
                long_benchmark.print_summary()

            except Exception as e:
                print(f"Error running long-context benchmark: {str(e)}")
                print("Note: This is expected if the system has limited GPU memory")
        else:
            print(
                "\\nSkipping long-context benchmark (use --skip-long-context-benchmark to disable this message)"
            )

        # Generate comparison plots
        self.plot_comparisons()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save experiment results to JSON."""
        results_file = self.output_dir / "experiment_results.json"
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\\nResults saved to: {results_file}")

    def log_comparison_metrics(self):
        """Log comparison metrics to wandb."""
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}

        if not valid_results:
            return

        # Create comparison table
        comparison_data = []
        for model_name, metrics in valid_results.items():
            comparison_data.append(
                {
                    "model": model_name,
                    "parameters_M": metrics["num_parameters"] / 1e6,
                    "training_time_min": metrics["training_time_seconds"] / 60,
                    "inference_time_ms": metrics["inference_time_ms"],
                    "val_perplexity": metrics["final_val_perplexity"]
                    if isinstance(metrics["final_val_perplexity"], (int, float))
                    and not torch.isinf(torch.tensor(metrics["final_val_perplexity"]))
                    else None,
                    "val_loss": metrics["final_val_loss"]
                    if torch.is_tensor(metrics["final_val_loss"])
                    else metrics["final_val_loss"],
                }
            )

        # Log comparison table
        wandb.log(
            {
                "model_comparison": wandb.Table(
                    columns=[
                        "model",
                        "parameters_M",
                        "training_time_min",
                        "inference_time_ms",
                        "val_perplexity",
                        "val_loss",
                    ],
                    data=[
                        [
                            row[col]
                            for col in [
                                "model",
                                "parameters_M",
                                "training_time_min",
                                "inference_time_ms",
                                "val_perplexity",
                                "val_loss",
                            ]
                        ]
                        for row in comparison_data
                    ],
                )
            }
        )

        # Log best model metrics
        valid_perplexities = {
            k: v["final_val_perplexity"]
            for k, v in valid_results.items()
            if isinstance(v["final_val_perplexity"], (int, float))
            and not torch.isinf(torch.tensor(v["final_val_perplexity"]))
        }

        if valid_perplexities:
            best_model = min(
                valid_perplexities.keys(), key=lambda x: valid_perplexities[x]
            )
            wandb.log(
                {
                    "best_model_by_perplexity": best_model,
                    "best_perplexity": valid_perplexities[best_model],
                }
            )

        fastest_model = min(
            valid_results.keys(), key=lambda x: valid_results[x]["inference_time_ms"]
        )
        wandb.log(
            {
                "fastest_model": fastest_model,
                "fastest_inference_time_ms": valid_results[fastest_model][
                    "inference_time_ms"
                ],
            }
        )

    def plot_comparisons(self):
        """Generate comparison plots."""
        if not self.results:
            return

        print("\\nGenerating comparison plots...")

        # Filter out error results
        valid_results = {k: v for k, v in self.results.items() if "error" not in v}

        if not valid_results:
            print("No valid results to plot.")
            return

        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        models = list(valid_results.keys())

        # Training time comparison
        train_times = [
            valid_results[model]["training_time_seconds"] / 60 for model in models
        ]  # Convert to minutes
        ax1.bar(models, train_times)
        ax1.set_title("Training Time Comparison")
        ax1.set_ylabel("Training Time (minutes)")
        ax1.tick_params(axis="x", rotation=45)

        # Inference time comparison
        inference_times = [
            valid_results[model]["inference_time_ms"] for model in models
        ]
        ax2.bar(models, inference_times)
        ax2.set_title("Inference Time Comparison")
        ax2.set_ylabel("Inference Time (ms)")
        ax2.tick_params(axis="x", rotation=45)

        # Final validation perplexity comparison
        val_perplexities = []
        for model in models:
            perplexity = valid_results[model]["final_val_perplexity"]
            if isinstance(perplexity, (int, float)) and not (
                torch.isinf(torch.tensor(perplexity))
                or torch.isnan(torch.tensor(perplexity))
            ):
                val_perplexities.append(perplexity)
            else:
                val_perplexities.append(float("inf"))

        # Only plot if we have valid perplexity values
        finite_perplexities = [
            p for p in val_perplexities if not torch.isinf(torch.tensor(p))
        ]
        if finite_perplexities:
            ax3.bar(models, val_perplexities)
            ax3.set_title("Final Validation Perplexity Comparison")
            ax3.set_ylabel("Perplexity")
            ax3.tick_params(axis="x", rotation=45)
        else:
            ax3.text(
                0.5,
                0.5,
                "No valid perplexity data",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )

        # Parameter count comparison
        param_counts = [
            valid_results[model]["num_parameters"] / 1e6 for model in models
        ]  # Convert to millions
        ax4.bar(models, param_counts)
        ax4.set_title("Model Size Comparison")
        ax4.set_ylabel("Parameters (millions)")
        ax4.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "comparison_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"Comparison plots saved to: {plot_file}")

        # Log plot to wandb if enabled
        if self.use_wandb:
            wandb.log({"comparison_plots": wandb.Image(str(plot_file))})

        plt.show()

    def print_summary(self):
        """Print experiment summary."""
        print("\\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        valid_results = {k: v for k, v in self.results.items() if "error" not in v}

        if not valid_results:
            print("No successful experiments to summarize.")
            return

        print(
            f"{'Model':<20} {'Params (M)':<12} {'Train Time (min)':<16} {'Inference (ms)':<15} {'Val Perplexity':<15}"
        )
        print("-" * 80)

        for model_name, metrics in valid_results.items():
            params = metrics["num_parameters"] / 1e6
            train_time = metrics["training_time_seconds"] / 60
            inference_time = metrics["inference_time_ms"]
            val_perplexity = metrics["final_val_perplexity"]

            # Format perplexity
            if isinstance(val_perplexity, (int, float)) and not (
                torch.isinf(torch.tensor(val_perplexity))
                or torch.isnan(torch.tensor(val_perplexity))
            ):
                perplexity_str = f"{val_perplexity:.2f}"
            else:
                perplexity_str = "N/A"

            print(
                f"{model_name:<20} {params:<12.2f} {train_time:<16.2f} {inference_time:<15.2f} {perplexity_str:<15}"
            )

        # Find best model by validation perplexity
        best_model = None
        best_perplexity = float("inf")

        for model_name, metrics in valid_results.items():
            perplexity = metrics["final_val_perplexity"]
            if isinstance(perplexity, (int, float)) and perplexity < best_perplexity:
                best_perplexity = perplexity
                best_model = model_name

        if best_model:
            print(
                f"\\nBest model by validation perplexity: {best_model} ({best_perplexity:.2f})"
            )

        # Find fastest model
        fastest_model = min(
            valid_results.keys(), key=lambda x: valid_results[x]["inference_time_ms"]
        )
        fastest_time = valid_results[fastest_model]["inference_time_ms"]
        print(f"Fastest inference: {fastest_model} ({fastest_time:.2f} ms)")

        # Print error summary if any
        error_results = {k: v for k, v in self.results.items() if "error" in v}
        if error_results:
            print(f"\\nModels with errors: {len(error_results)}")
            for model_name, error_info in error_results.items():
                print(f"  {model_name}: {error_info['error']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train and compare attention mechanisms"
    )

    # Experiment parameters
    parser.add_argument(
        "--output-dir",
        default="experiment_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training",
    )

    # Model parameters
    parser.add_argument("--vocab-size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--d-model", type=int, default=512, help="Model dimension")
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument(
        "--max-seq-len", type=int, default=1024, help="Maximum sequence length"
    )

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--train-samples", type=int, default=5000, help="Number of training samples"
    )
    parser.add_argument(
        "--val-samples", type=int, default=1000, help="Number of validation samples"
    )
    parser.add_argument(
        "--seq-len", type=int, default=512, help="Sequence length for training"
    )

    # Quick test mode
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced parameters",
    )

    # Text data options
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic data instead of real text (default: use real text)",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Tokenizer to use for real text (default: gpt2)",
    )

    # Wandb options
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for experiment tracking",
    )
    parser.add_argument(
        "--wandb-project",
        default="attention-comparison",
        help="Wandb project name (default: attention-comparison)",
    )

    # Benchmark options
    parser.add_argument(
        "--skip-long-context-benchmark",
        action="store_true",
        help="Skip the long-context benchmark (useful for limited memory systems)",
    )
    parser.add_argument(
        "--max-benchmark-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length for long-context benchmark (default: 4096)",
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig()

    # Handle text data options
    config.use_synthetic = (
        args.use_synthetic
    )  # Default to False (real text) unless --use-synthetic is passed

    config.tokenizer_name = args.tokenizer_name

    # Update config with command line arguments
    if args.quick_test:
        print("Running in quick test mode...")
        config.d_model = 128
        config.num_heads = 4
        config.num_layers = 2
        config.max_epochs = 2
        config.train_samples = 200
        config.val_samples = 50
        config.seq_len = 128
        config.max_seq_len = 256
    else:
        # Only set vocab_size if using synthetic data
        if config.use_synthetic:
            config.vocab_size = args.vocab_size
        config.d_model = args.d_model
        config.num_heads = args.num_heads
        config.num_layers = args.num_layers
        config.max_seq_len = args.max_seq_len
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_epochs = args.max_epochs
        config.train_samples = args.train_samples
        config.val_samples = args.val_samples
        config.seq_len = args.seq_len

    # Print configuration
    print("Experiment Configuration:")
    print(f"  Data type: {'Synthetic' if config.use_synthetic else 'Real text'}")
    if not config.use_synthetic:
        print(f"  Tokenizer: {config.tokenizer_name}")
    else:
        print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Model: {config.d_model}d, {config.num_heads}h, {config.num_layers}l")
    print(f"  Training: {config.max_epochs} epochs, {config.train_samples} samples")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Wandb logging: {'Enabled' if args.use_wandb else 'Disabled'}")

    # Create and run experiment
    runner = ExperimentRunner(
        config,
        args.output_dir,
        args.device,
        args.use_wandb,
        args.wandb_project,
        args.skip_long_context_benchmark,
        args.max_benchmark_seq_len,
    )
    runner.run_experiments()


if __name__ == "__main__":
    main()
