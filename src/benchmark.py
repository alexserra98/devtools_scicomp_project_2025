import torch
import torch.nn.functional as F
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import json
import os
import math
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

from src.attention_implementations import (
    StandardAttention,
    SparseAttention,
    FlashAttention,
)
from src.transformer_model import TransformerLM, count_parameters, get_model_size_mb


class TextDataset(Dataset):
    """Real text dataset for training language models."""

    def __init__(
        self,
        tokenizer,
        seq_len: int,
        num_samples: int,
        dataset_name: str = "wikitext",
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Load dataset
        if dataset_name == "wikitext":
            try:
                from datasets import load_dataset

                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
                texts = [item["text"] for item in dataset if item["text"].strip()]
            except Exception as e:
                # Fallback to simple text if datasets not available
                print(
                    f"Warning: Could not load wikitext dataset ({e}), using simple text generation"
                )
                texts = self._generate_simple_texts()
        else:
            texts = self._generate_simple_texts()

        # Tokenize and prepare sequences
        self.data = self._prepare_sequences(texts)

    def _generate_simple_texts(self) -> List[str]:
        """Generate simple text samples as fallback."""
        simple_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have revolutionized computer vision and NLP.",
            "Transformers are a type of neural network architecture.",
            "Attention mechanisms allow models to focus on relevant parts of input.",
            "Large language models can generate human-like text.",
            "Training neural networks requires large amounts of data.",
            "Gradient descent is an optimization algorithm used in machine learning.",
            "Backpropagation is used to compute gradients in neural networks.",
        ]

        # Repeat and extend to get enough text
        extended_texts = []
        for i in range(self.num_samples // len(simple_texts) + 1):
            for text in simple_texts:
                extended_texts.append(
                    f"{text} This is sentence {i + 1} in the training corpus."
                )

        return extended_texts[: self.num_samples * 2]  # Extra to account for filtering

    def _prepare_sequences(self, texts: List[str]) -> torch.Tensor:
        """Tokenize texts and create fixed-length sequences."""
        # Concatenate all texts
        full_text = " ".join(texts)

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Create sequences of fixed length
        sequences = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len // 2):  # 50% overlap
            sequence = tokens[i : i + self.seq_len]
            if len(sequence) == self.seq_len:
                sequences.append(sequence)

            if len(sequences) >= self.num_samples:
                break

        # If we don't have enough sequences, repeat some
        while len(sequences) < self.num_samples:
            sequences.extend(
                sequences[: min(len(sequences), self.num_samples - len(sequences))]
            )

        return torch.tensor(sequences[: self.num_samples], dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


class SyntheticTextDataset(Dataset):
    """Synthetic text dataset for benchmarking (kept for backward compatibility)."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

        # Generate random sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.results = {}

    def benchmark_attention_mechanisms(
        self,
        seq_lengths: List[int] = [128, 256, 512, 1024, 2048],
        d_model: int = 512,
        num_heads: int = 8,
        batch_size: int = 4,
        num_iterations: int = 50,
    ) -> Dict[str, Dict[str, List[float]]]:
        """Benchmark different attention mechanisms across sequence lengths."""

        print("Benchmarking attention mechanisms...")
        results = {
            "standard": {"runtime": [], "memory": [], "success": []},
            "sparse_local": {"runtime": [], "memory": [], "success": []},
            "sparse_strided": {"runtime": [], "memory": [], "success": []},
            "flash": {"runtime": [], "memory": [], "success": []},
        }

        for seq_len in tqdm(seq_lengths, desc="Sequence lengths"):
            print(f"\\nTesting sequence length: {seq_len}")

            # Test each attention mechanism with error handling
            attention_configs = [
                ("standard", StandardAttention(d_model, num_heads)),
                (
                    "sparse_local",
                    SparseAttention(
                        d_model,
                        num_heads,
                        sparsity_pattern="local",
                        window_size=min(256, seq_len // 4),
                    ),
                ),
                (
                    "sparse_strided",
                    SparseAttention(
                        d_model,
                        num_heads,
                        sparsity_pattern="strided",
                        window_size=min(128, seq_len // 8),
                        stride=max(
                            1, min(32, seq_len // 32)
                        ),  # Ensure stride is at least 1
                    ),
                ),
                (
                    "flash",
                    FlashAttention(
                        d_model, num_heads, block_size=min(256, seq_len // 4)
                    ),
                ),
            ]

            for name, attention in attention_configs:
                try:
                    attention = attention.to(self.device)
                    runtime, memory = self._benchmark_single_attention(
                        attention, batch_size, seq_len, d_model, num_iterations
                    )
                    results[name]["runtime"].append(runtime)
                    results[name]["memory"].append(memory)
                    results[name]["success"].append(True)
                    print(f"  {name:15}: {runtime:7.2f}ms, {memory:7.1f}MB")

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        results[name]["runtime"].append(float("inf"))
                        results[name]["memory"].append(float("inf"))
                        results[name]["success"].append(False)
                        print(f"  {name:15}: OOM (Out of Memory)")
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise e
                except Exception as e:
                    print(f"  {name:15}: Error - {str(e)}")
                    results[name]["runtime"].append(float("inf"))
                    results[name]["memory"].append(float("inf"))
                    results[name]["success"].append(False)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            # Clear cache between tests
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.results["attention_benchmark"] = {
            "seq_lengths": seq_lengths,
            "results": results,
        }

        return results

    def _benchmark_single_attention(
        self,
        attention_layer: torch.nn.Module,
        batch_size: int,
        seq_len: int,
        d_model: int,
        num_iterations: int,
    ) -> Tuple[float, float]:
        """Benchmark a single attention layer."""

        # Create input on the same device as the attention layer
        x = torch.randn(batch_size, seq_len, d_model, device=self.device)

        # Custom benchmark function that uses the correct device
        import time

        # Warmup
        for _ in range(10):
            _ = attention_layer(x)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            _ = attention_layer(x)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (
            (end_time - start_time) / num_iterations * 1000
        )  # Convert to milliseconds

        # Get memory usage
        peak_memory = 0
        if self.device.type == "cuda":
            peak_memory = (
                torch.cuda.max_memory_allocated() / 1024 / 1024
            )  # Convert to MB
        else:
            # For CPU, estimate memory based on tensor size
            peak_memory = (
                x.numel() * x.element_size() * 4 / 1024 / 1024
            )  # Rough estimate

        return avg_time, peak_memory

    def benchmark_full_models(
        self,
        vocab_size: int = 10000,
        seq_lengths: List[int] = [512, 1024, 2048],
        batch_size: int = 4,
        num_steps: int = 100,
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark full transformer models with different attention mechanisms across multiple sequence lengths."""

        print("Benchmarking full transformer models...")

        model_configs = {
            "standard": {"attention_type": "standard"},
            "sparse_local": {
                "attention_type": "sparse",
                "sparsity_pattern": "local",
                "window_size": 64,
            },
            "flash": {"attention_type": "flash", "block_size": 64},
        }

        results = {}

        for model_name, config in model_configs.items():
            print(f"\\nBenchmarking {model_name} model across sequence lengths...")
            model_results = {
                "seq_lengths": seq_lengths,
                "train_time_ms": [],
                "train_memory_mb": [],
                "inference_time_ms": [],
                "inference_memory_mb": [],
                "success": [],
                "num_parameters": None,
                "model_size_mb": None,
            }

            for seq_len in seq_lengths:
                print(f"  Testing sequence length: {seq_len}")

                try:
                    # Adjust window size based on sequence length for sparse attention
                    adjusted_config = config.copy()
                    if "window_size" in adjusted_config:
                        adjusted_config["window_size"] = min(
                            adjusted_config["window_size"], seq_len // 4
                        )
                    if "block_size" in adjusted_config:
                        adjusted_config["block_size"] = min(
                            adjusted_config["block_size"], seq_len // 4
                        )

                    # Create model
                    model = TransformerLM(
                        vocab_size=vocab_size,
                        d_model=256,
                        num_heads=8,
                        num_layers=4,
                        max_seq_len=seq_len,
                        **adjusted_config,
                    ).to(self.device)

                    # Store model info once
                    if model_results["num_parameters"] is None:
                        model_results["num_parameters"] = count_parameters(model)
                        model_results["model_size_mb"] = get_model_size_mb(model)

                    # Create synthetic data
                    dataset = SyntheticTextDataset(
                        vocab_size, seq_len, batch_size * num_steps
                    )
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=False
                    )

                    # Benchmark training
                    train_time, train_memory = self._benchmark_training(
                        model, dataloader, num_steps
                    )

                    # Benchmark inference
                    inference_time, inference_memory = self._benchmark_inference(
                        model, batch_size, seq_len, num_steps
                    )

                    model_results["train_time_ms"].append(train_time)
                    model_results["train_memory_mb"].append(train_memory)
                    model_results["inference_time_ms"].append(inference_time)
                    model_results["inference_memory_mb"].append(inference_memory)
                    model_results["success"].append(True)

                    print(
                        f"    Train: {train_time:.1f}ms, {train_memory:.1f}MB | "
                        f"Inference: {inference_time:.1f}ms, {inference_memory:.1f}MB"
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        model_results["train_time_ms"].append(float("inf"))
                        model_results["train_memory_mb"].append(float("inf"))
                        model_results["inference_time_ms"].append(float("inf"))
                        model_results["inference_memory_mb"].append(float("inf"))
                        model_results["success"].append(False)
                        print("    OOM (Out of Memory)")
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                    else:
                        raise e
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    model_results["train_time_ms"].append(float("inf"))
                    model_results["train_memory_mb"].append(float("inf"))
                    model_results["inference_time_ms"].append(float("inf"))
                    model_results["inference_memory_mb"].append(float("inf"))
                    model_results["success"].append(False)
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                finally:
                    # Clean up
                    try:
                        del model
                    except Exception:
                        pass
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            results[model_name] = model_results

        self.results["model_benchmark"] = results
        return results

    def _benchmark_training(
        self, model: TransformerLM, dataloader: DataLoader, num_steps: int
    ) -> Tuple[float, float]:
        """Benchmark model training."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()

        for i, batch in enumerate(dataloader):
            if i >= num_steps:
                break

            input_ids = batch["input_ids"].to(self.device)

            optimizer.zero_grad()
            outputs = model(input_ids, targets=input_ids)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

            if self.device.type == "cuda":
                torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_steps * 1000  # ms per step

        peak_memory = 0
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        return avg_time, peak_memory

    def _benchmark_inference(
        self, model: TransformerLM, batch_size: int, seq_len: int, num_steps: int
    ) -> Tuple[float, float]:
        """Benchmark model inference."""
        model.eval()

        input_ids = torch.randint(
            0, model.vocab_size, (batch_size, seq_len), device=self.device
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_ids)

        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_steps):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                _ = model(input_ids)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.time()
        avg_time = (end_time - start_time) / num_steps * 1000  # ms per step

        peak_memory = 0
        if self.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        return avg_time, peak_memory

    def plot_results(self, save_dir: Optional[str] = None):
        """Plot benchmark results."""
        if "attention_benchmark" in self.results:
            self._plot_attention_benchmark(save_dir)

        if "model_benchmark" in self.results:
            self._plot_model_benchmark(save_dir)

    def _plot_attention_benchmark(self, save_dir: Optional[str] = None):
        """Plot attention mechanism benchmark results."""
        data = self.results["attention_benchmark"]
        seq_lengths = data["seq_lengths"]
        results = data["results"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Runtime plot
        for attention_type, metrics in results.items():
            # Filter out inf values for plotting
            valid_points = [
                (s, r)
                for s, r, success in zip(
                    seq_lengths, metrics["runtime"], metrics["success"]
                )
                if success and r != float("inf")
            ]
            if valid_points:
                valid_seq, valid_runtime = zip(*valid_points)
                ax1.plot(
                    valid_seq,
                    valid_runtime,
                    marker="o",
                    label=attention_type.replace("_", " ").title(),
                )

        ax1.set_xlabel("Sequence Length")
        ax1.set_ylabel("Runtime (ms)")
        ax1.set_title("Attention Mechanism Runtime vs Sequence Length")
        ax1.legend()
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)

        # Memory plot
        for attention_type, metrics in results.items():
            # Filter out inf values for plotting
            valid_points = [
                (s, m)
                for s, m, success in zip(
                    seq_lengths, metrics["memory"], metrics["success"]
                )
                if success and m != float("inf")
            ]
            if valid_points:
                valid_seq, valid_memory = zip(*valid_points)
                ax2.plot(
                    valid_seq,
                    valid_memory,
                    marker="s",
                    label=attention_type.replace("_", " ").title(),
                )

        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Peak Memory (MB)")
        ax2.set_title("Attention Mechanism Memory Usage vs Sequence Length")
        ax2.legend()
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "attention_benchmark.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def _plot_model_benchmark(self, save_dir: Optional[str] = None):
        """Plot full model benchmark results."""
        data = self.results["model_benchmark"]

        models = list(data.keys())

        # Check if we have sequence length data (new format) or single values (old format)
        first_model = data[models[0]]
        if "seq_lengths" in first_model:
            # New format with multiple sequence lengths
            self._plot_model_benchmark_multi_seq(data, save_dir)
        else:
            # Old format with single values
            self._plot_model_benchmark_single(data, save_dir)

    def _plot_model_benchmark_multi_seq(
        self, data: Dict, save_dir: Optional[str] = None
    ):
        """Plot model benchmark results for multiple sequence lengths."""
        models = list(data.keys())
        seq_lengths = data[models[0]]["seq_lengths"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()

        metrics = [
            "train_time_ms",
            "inference_time_ms",
            "train_memory_mb",
            "inference_memory_mb",
        ]
        titles = [
            "Training Time",
            "Inference Time",
            "Training Memory",
            "Inference Memory",
        ]
        y_labels = ["Time (ms)", "Time (ms)", "Memory (MB)", "Memory (MB)"]

        for i, (metric, title, ylabel) in enumerate(zip(metrics, titles, y_labels)):
            for model in models:
                model_data = data[model]
                # Filter out inf values and failed runs
                valid_points = [
                    (s, m)
                    for s, m, success in zip(
                        seq_lengths, model_data[metric], model_data["success"]
                    )
                    if success and m != float("inf")
                ]
                if valid_points:
                    valid_seq, valid_metric = zip(*valid_points)
                    axes[i].plot(
                        valid_seq,
                        valid_metric,
                        marker="o",
                        label=model.replace("_", " ").title(),
                    )

            axes[i].set_xlabel("Sequence Length")
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f"{title} vs Sequence Length")
            axes[i].legend()
            axes[i].set_xscale("log", base=2)
            axes[i].set_yscale("log")
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "model_benchmark.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def _plot_model_benchmark_single(self, data: Dict, save_dir: Optional[str] = None):
        """Plot model benchmark results for single sequence length (backward compatibility)."""
        models = list(data.keys())
        metrics = [
            "train_time_ms",
            "inference_time_ms",
            "train_memory_mb",
            "inference_memory_mb",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            values = [data[model][metric] for model in models]
            axes[i].bar(models, values)
            axes[i].set_title(metric.replace("_", " ").title())
            axes[i].set_ylabel(metric.split("_")[-1].upper())

            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "model_benchmark.png"),
                dpi=300,
                bbox_inches="tight",
            )

        plt.show()

    def save_results(self, filepath: str):
        """Save benchmark results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert inf values to string for JSON serialization
        def convert_inf(obj):
            if isinstance(obj, dict):
                return {k: convert_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_inf(v) for v in obj]
            elif obj == float("inf"):
                return "inf"
            elif obj == float("-inf"):
                return "-inf"
            else:
                return obj

        serializable_results = convert_inf(self.results)

        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)

    def load_results(self, filepath: str):
        """Load benchmark results from JSON file."""

        def restore_inf(obj):
            if isinstance(obj, dict):
                return {k: restore_inf(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [restore_inf(v) for v in obj]
            elif obj == "inf":
                return float("inf")
            elif obj == "-inf":
                return float("-inf")
            else:
                return obj

        with open(filepath, "r") as f:
            loaded_results = json.load(f)
            self.results = restore_inf(loaded_results)


class PerplexityEvaluator:
    """Evaluate and compare perplexity across different attention mechanisms."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def compare_perplexity(
        self,
        vocab_size: int = 10000,
        seq_len: int = 512,
        batch_size: int = 8,
        num_epochs: int = 5,
        num_samples: int = 1000,
    ) -> Dict[str, Dict[str, float]]:
        """Compare perplexity across different attention mechanisms."""

        print("Comparing perplexity across attention mechanisms...")

        model_configs = {
            "standard": {"attention_type": "standard"},
            "sparse_local": {
                "attention_type": "sparse",
                "sparsity_pattern": "local",
                "window_size": 64,
            },
            "flash": {"attention_type": "flash", "block_size": 64},
        }

        results = {}

        # Create datasets
        train_dataset = SyntheticTextDataset(vocab_size, seq_len, num_samples)
        val_dataset = SyntheticTextDataset(vocab_size, seq_len, num_samples // 4)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for model_name, config in model_configs.items():
            print(f"\\nTraining {model_name} model...")

            # Create model
            model = TransformerLM(
                vocab_size=vocab_size,
                d_model=256,
                num_heads=8,
                num_layers=4,
                max_seq_len=seq_len,
                learning_rate=5e-4,
                **config,
            )

            # Train model
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                accelerator="gpu" if self.device.type == "cuda" else "cpu",
                devices=1,
                enable_progress_bar=True,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
            )

            trainer.fit(model, train_loader, val_loader)

            # Evaluate final perplexity
            model.eval()
            total_loss = 0
            total_tokens = 0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    outputs = model(input_ids, targets=input_ids)
                    loss = outputs["loss"]

                    total_loss += loss.item() * input_ids.numel()
                    total_tokens += input_ids.numel()

            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)

            results[model_name] = {
                "final_loss": avg_loss,
                "final_perplexity": perplexity,
                "num_parameters": count_parameters(model),
            }

            print(f"{model_name}: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.2f}")

        return results


def run_comprehensive_benchmark(
    output_dir: str = "benchmark_results", device: str = "auto"
):
    """Run comprehensive benchmark suite and save results."""

    print("Starting comprehensive benchmark suite...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize benchmark
    benchmark = PerformanceBenchmark(device)

    # Run attention mechanism benchmarks
    print("\\n" + "=" * 50)
    print("ATTENTION MECHANISM BENCHMARKS (Short to Long Context)")
    print("=" * 50)

    attention_results = benchmark.benchmark_attention_mechanisms(
        seq_lengths=[128, 256, 512, 1024, 2048, 4096, 8192],
        d_model=512,
        num_heads=8,
        batch_size=4,
        num_iterations=30,
    )

    # Run full model benchmarks
    print("\\n" + "=" * 50)
    print("FULL MODEL BENCHMARKS (Multiple Sequence Lengths)")
    print("=" * 50)

    model_results = benchmark.benchmark_full_models(
        vocab_size=10000,
        seq_lengths=[512, 1024, 2048, 4096],
        batch_size=4,
        num_steps=50,
    )

    # Plot and save results
    benchmark.plot_results(output_dir)
    benchmark.save_results(os.path.join(output_dir, "benchmark_results.json"))

    # Run perplexity comparison
    print("\\n" + "=" * 50)
    print("PERPLEXITY COMPARISON")
    print("=" * 50)

    evaluator = PerplexityEvaluator(device)
    perplexity_results = evaluator.compare_perplexity(
        vocab_size=5000, seq_len=256, batch_size=8, num_epochs=3, num_samples=500
    )

    # Save perplexity results
    with open(os.path.join(output_dir, "perplexity_results.json"), "w") as f:
        json.dump(perplexity_results, f, indent=2)

    # Print summary
    print("\\n" + "=" * 50)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 50)

    # Attention mechanism summary
    attention_data = benchmark.results["attention_benchmark"]
    seq_lengths = attention_data["seq_lengths"]
    attention_results = attention_data["results"]

    print("\\n📊 ATTENTION MECHANISM SCALING:")
    print("Sequence Length Performance (Runtime | Memory):")

    for seq_len in seq_lengths:
        seq_idx = seq_lengths.index(seq_len)
        print(f"\\n  {seq_len:5d} tokens:")

        for attention_type, metrics in attention_results.items():
            if seq_idx < len(metrics["success"]) and metrics["success"][seq_idx]:
                runtime = metrics["runtime"][seq_idx]
                memory = metrics["memory"][seq_idx]
                print(f"    {attention_type:15}: {runtime:7.1f}ms | {memory:7.1f}MB")
            else:
                print(f"    {attention_type:15}: {'OOM':>7s} | {'OOM':>7s}")

    # Model performance summary
    print("\\n🏗️  FULL MODEL SCALING:")
    for model_name, metrics in model_results.items():
        print(f"\\n  {model_name.replace('_', ' ').title()} Model:")
        print(f"    Parameters: {metrics['num_parameters']:,}")

        if "seq_lengths" in metrics:
            seq_lengths = metrics["seq_lengths"]
            print("    Performance by sequence length:")

            for i, seq_len in enumerate(seq_lengths):
                if i < len(metrics["success"]) and metrics["success"][i]:
                    train_time = metrics["train_time_ms"][i]
                    train_mem = metrics["train_memory_mb"][i]
                    inf_time = metrics["inference_time_ms"][i]
                    inf_mem = metrics["inference_memory_mb"][i]

                    print(
                        f"      {seq_len:5d} tokens: Train {train_time:6.1f}ms/{train_mem:5.0f}MB | "
                        f"Inference {inf_time:6.1f}ms/{inf_mem:5.0f}MB"
                    )
                else:
                    print(f"      {seq_len:5d} tokens: OOM (Out of Memory)")
        else:
            # Backward compatibility for old format
            print(
                f"    Training: {metrics['train_time_ms']:.2f}ms/step, {metrics['train_memory_mb']:.1f}MB"
            )
            print(
                f"    Inference: {metrics['inference_time_ms']:.2f}ms/step, {metrics['inference_memory_mb']:.1f}MB"
            )

    # Key insights
    print("\\n🔍 KEY INSIGHTS:")

    # Find where each attention mechanism starts failing
    max_working_lengths = {}
    for attention_type, metrics in attention_results.items():
        max_len = 0
        for i, success in enumerate(metrics["success"]):
            if success:
                max_len = seq_lengths[i]
        max_working_lengths[attention_type] = max_len

    print("   Maximum working sequence lengths:")
    for attention_type, max_len in max_working_lengths.items():
        if max_len > 0:
            print(f"     {attention_type:15}: {max_len:5d} tokens")
        else:
            print(f"     {attention_type:15}: Failed on all lengths")

    # Memory efficiency comparison at longest common length
    common_lengths = [
        seq_len
        for seq_len in seq_lengths
        if all(
            attention_results[att]["success"][seq_lengths.index(seq_len)]
            for att in attention_results.keys()
        )
    ]

    if common_lengths:
        longest_common = max(common_lengths)
        common_idx = seq_lengths.index(longest_common)
        print(f"\\n   Memory usage at {longest_common} tokens:")

        memories = [
            (att, attention_results[att]["memory"][common_idx])
            for att in attention_results.keys()
        ]
        memories.sort(key=lambda x: x[1])

        for att, mem in memories:
            print(f"     {att:15}: {mem:7.1f} MB")

    print("\\nPerplexity Comparison:")
    for model_name, metrics in perplexity_results.items():
        print(f"  {model_name}: {metrics['final_perplexity']:.2f}")

    print(f"\\nResults saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run attention mechanism benchmarks")
    parser.add_argument(
        "--output-dir", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use",
    )

    args = parser.parse_args()

    run_comprehensive_benchmark(args.output_dir, args.device)
