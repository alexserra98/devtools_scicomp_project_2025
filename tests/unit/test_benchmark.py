"""
Unit tests for benchmark module components.

Tests cover:
- SyntheticTextDataset: Dataset creation and data loading
- TextDataset: Real text dataset functionality
- PerformanceBenchmark: Benchmarking infrastructure
- PerplexityEvaluator: Model evaluation functionality
"""

import unittest
import torch
import tempfile
import os
from unittest.mock import patch, MagicMock

try:
    from src.benchmark import (
        SyntheticTextDataset,
        TextDataset,
        PerformanceBenchmark,
        PerplexityEvaluator,
    )
    from src.attention_implementations import StandardAttention
except ImportError:
    # Fallback for direct test execution
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from benchmark import (
        SyntheticTextDataset,
        TextDataset,
        PerformanceBenchmark,
        PerplexityEvaluator,
    )
    from attention_implementations import StandardAttention


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def encode(self, text, add_special_tokens=True):
        # Simple mock: convert text to list of token IDs based on length
        tokens = []
        words = text.split()
        for word in words:
            # Generate deterministic token IDs based on word
            for char in word:
                tokens.append(ord(char) % self.vocab_size)
        return tokens[:100]  # Limit length for testing


class TestSyntheticTextDataset(unittest.TestCase):
    """Test suite for SyntheticTextDataset."""

    def test_initialization(self):
        """Test dataset initialization."""
        vocab_size = 1000
        seq_len = 64
        num_samples = 100

        dataset = SyntheticTextDataset(vocab_size, seq_len, num_samples)

        self.assertEqual(dataset.vocab_size, vocab_size)
        self.assertEqual(dataset.seq_len, seq_len)
        self.assertEqual(dataset.num_samples, num_samples)
        self.assertEqual(len(dataset), num_samples)
        self.assertEqual(dataset.data.shape, (num_samples, seq_len))

    def test_getitem(self):
        """Test dataset item retrieval."""
        dataset = SyntheticTextDataset(100, 32, 50)

        item = dataset[0]

        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (32,))
        self.assertTrue((item["input_ids"] >= 0).all())
        self.assertTrue((item["input_ids"] < 100).all())

    def test_different_sizes(self):
        """Test dataset with different configurations."""
        configs = [(50, 16, 10), (1000, 128, 100), (5000, 256, 200)]

        for vocab_size, seq_len, num_samples in configs:
            dataset = SyntheticTextDataset(vocab_size, seq_len, num_samples)

            self.assertEqual(len(dataset), num_samples)
            item = dataset[0]
            self.assertEqual(item["input_ids"].shape, (seq_len,))
            self.assertTrue((item["input_ids"] < vocab_size).all())


class TestTextDataset(unittest.TestCase):
    """Test suite for TextDataset."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_tokenizer = MockTokenizer()

    @patch("benchmark.load_dataset")
    def test_initialization_with_wikitext(self, mock_load_dataset):
        """Test dataset initialization with WikiText."""
        # Mock the dataset
        mock_dataset = [
            {"text": "This is a test sentence."},
            {"text": "Another test sentence here."},
            {"text": "And yet another one."},
        ]
        mock_load_dataset.return_value = mock_dataset

        dataset = TextDataset(
            tokenizer=self.mock_tokenizer,
            seq_len=32,
            num_samples=10,
            dataset_name="wikitext",
        )

        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.seq_len, 32)
        self.assertEqual(dataset.num_samples, 10)

    def test_initialization_fallback(self):
        """Test dataset initialization with fallback text."""
        # This will trigger the fallback to simple text generation
        dataset = TextDataset(
            tokenizer=self.mock_tokenizer,
            seq_len=32,
            num_samples=10,
            dataset_name="nonexistent",
        )

        self.assertEqual(len(dataset), 10)
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertEqual(item["input_ids"].shape, (32,))

    def test_simple_text_generation(self):
        """Test the fallback simple text generation."""
        dataset = TextDataset(
            tokenizer=self.mock_tokenizer,
            seq_len=16,
            num_samples=5,
            dataset_name="simple",  # Will trigger fallback
        )

        # Should create valid sequences
        self.assertEqual(len(dataset), 5)

        for i in range(len(dataset)):
            item = dataset[i]
            self.assertEqual(item["input_ids"].shape, (16,))
            self.assertTrue((item["input_ids"] >= 0).all())

    def test_sequence_preparation(self):
        """Test sequence preparation from texts."""
        # Create a dataset with known text
        texts = [
            "Short text.",
            "Longer text with more words to test sequence creation.",
        ]

        dataset = TextDataset.__new__(TextDataset)  # Create without __init__
        dataset.tokenizer = self.mock_tokenizer
        dataset.seq_len = 10
        dataset.num_samples = 5

        sequences = dataset._prepare_sequences(texts)

        self.assertIsInstance(sequences, torch.Tensor)
        self.assertEqual(sequences.dtype, torch.long)
        # Should create some sequences
        self.assertGreater(sequences.shape[0], 0)
        self.assertEqual(sequences.shape[1], 10)


class TestPerformanceBenchmark(unittest.TestCase):
    """Test suite for PerformanceBenchmark."""

    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = PerformanceBenchmark(device="cpu")

    def test_initialization(self):
        """Test benchmark initialization."""
        self.assertEqual(self.benchmark.device.type, "cpu")
        self.assertIsInstance(self.benchmark.results, dict)

    def test_benchmark_single_attention(self):
        """Test benchmarking a single attention mechanism."""
        attention = StandardAttention(d_model=64, num_heads=4)

        runtime, memory = self.benchmark._benchmark_single_attention(
            attention, batch_size=2, seq_len=32, d_model=64, num_iterations=5
        )

        self.assertIsInstance(runtime, float)
        self.assertIsInstance(memory, float)
        self.assertGreater(runtime, 0)
        self.assertGreater(memory, 0)

    def test_benchmark_attention_mechanisms(self):
        """Test benchmarking multiple attention mechanisms."""
        # Use small parameters for fast testing
        results = self.benchmark.benchmark_attention_mechanisms(
            seq_lengths=[16, 32],
            d_model=64,
            num_heads=4,
            batch_size=2,
            num_iterations=3,
        )

        # Check result structure
        self.assertIn("standard", results)

        for attention_type, metrics in results.items():
            self.assertIn("runtime", metrics)
            self.assertIn("memory", metrics)
            self.assertIn("success", metrics)

            # Should have results for each sequence length
            self.assertEqual(len(metrics["runtime"]), 2)
            self.assertEqual(len(metrics["memory"]), 2)
            self.assertEqual(len(metrics["success"]), 2)

    def test_benchmark_full_models(self):
        """Test benchmarking full models."""
        # Use very small parameters for fast testing
        results = self.benchmark.benchmark_full_models(
            vocab_size=100, seq_lengths=[16, 32], batch_size=2, num_steps=3
        )

        # Check that we have results for different model types
        self.assertIn("standard", results)

        for model_name, metrics in results.items():
            self.assertIn("seq_lengths", metrics)
            self.assertIn("train_time_ms", metrics)
            self.assertIn("inference_time_ms", metrics)
            self.assertIn("num_parameters", metrics)

            # Should have results for each sequence length
            self.assertEqual(len(metrics["train_time_ms"]), 2)
            self.assertEqual(len(metrics["inference_time_ms"]), 2)

    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        # Create some mock results
        self.benchmark.results = {
            "test_benchmark": {
                "runtime": [1.0, 2.0, float("inf")],
                "memory": [10.0, 20.0, 30.0],
                "success": [True, True, False],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save results
            self.benchmark.save_results(filepath)

            # Load results into new benchmark instance
            new_benchmark = PerformanceBenchmark(device="cpu")
            new_benchmark.load_results(filepath)

            # Check that results are preserved
            self.assertEqual(
                new_benchmark.results["test_benchmark"]["runtime"],
                [1.0, 2.0, float("inf")],
            )
            self.assertEqual(
                new_benchmark.results["test_benchmark"]["success"], [True, True, False]
            )
        finally:
            os.unlink(filepath)


class TestPerplexityEvaluator(unittest.TestCase):
    """Test suite for PerplexityEvaluator."""

    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = PerplexityEvaluator(device="cpu")

    def test_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.device.type, "cpu")

    @patch("benchmark.pl.Trainer")
    def test_compare_perplexity(self, mock_trainer_class):
        """Test perplexity comparison functionality."""
        # Mock the trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        # Use very small parameters for testing
        results = self.evaluator.compare_perplexity(
            vocab_size=100, seq_len=32, batch_size=2, num_epochs=1, num_samples=10
        )

        # Check that we get results for different models
        self.assertIn("standard", results)

        for model_name, metrics in results.items():
            self.assertIn("final_loss", metrics)
            self.assertIn("final_perplexity", metrics)
            self.assertIn("num_parameters", metrics)

            self.assertIsInstance(metrics["final_loss"], float)
            self.assertIsInstance(metrics["final_perplexity"], float)
            self.assertIsInstance(metrics["num_parameters"], int)


class TestBenchmarkIntegration(unittest.TestCase):
    """Integration tests for benchmark components."""

    def test_end_to_end_small_benchmark(self):
        """Test a complete small-scale benchmark run."""
        # This test runs a very minimal benchmark to ensure all components work together
        benchmark = PerformanceBenchmark(device="cpu")

        # Run a minimal attention benchmark
        attention_results = benchmark.benchmark_attention_mechanisms(
            seq_lengths=[8, 16], d_model=32, num_heads=2, batch_size=1, num_iterations=2
        )

        # Verify we got some results
        self.assertIn("standard", attention_results)

        # Check that results have the expected structure
        standard_results = attention_results["standard"]
        self.assertEqual(len(standard_results["runtime"]), 2)
        self.assertEqual(len(standard_results["memory"]), 2)
        self.assertEqual(len(standard_results["success"]), 2)

        # All runs should succeed with these small parameters
        self.assertTrue(all(standard_results["success"]))

    def test_dataset_compatibility(self):
        """Test that datasets work with DataLoader."""
        from torch.utils.data import DataLoader

        # Test SyntheticTextDataset
        synthetic_dataset = SyntheticTextDataset(
            vocab_size=100, seq_len=16, num_samples=20
        )
        synthetic_loader = DataLoader(synthetic_dataset, batch_size=4, shuffle=True)

        batch = next(iter(synthetic_loader))
        self.assertIn("input_ids", batch)
        self.assertEqual(batch["input_ids"].shape, (4, 16))

        # Test TextDataset
        mock_tokenizer = MockTokenizer(vocab_size=100)
        text_dataset = TextDataset(
            tokenizer=mock_tokenizer, seq_len=16, num_samples=20, dataset_name="simple"
        )
        text_loader = DataLoader(text_dataset, batch_size=4, shuffle=False)

        batch = next(iter(text_loader))
        self.assertIn("input_ids", batch)
        self.assertEqual(batch["input_ids"].shape, (4, 16))


class TestBenchmarkErrorHandling(unittest.TestCase):
    """Test error handling in benchmark components."""

    def test_benchmark_with_invalid_attention(self):
        """Test benchmark behavior with problematic attention mechanisms."""
        benchmark = PerformanceBenchmark(device="cpu")

        # Create an attention that might cause issues
        class ProblematicAttention(torch.nn.Module):
            def forward(self, x, mask=None):
                # This will cause an out-of-memory error for large inputs
                return torch.zeros(1000000, 1000000, 1000000)  # Huge tensor

        problematic_attention = ProblematicAttention()

        # This should handle the error gracefully
        try:
            runtime, memory = benchmark._benchmark_single_attention(
                problematic_attention,
                batch_size=1,
                seq_len=16,
                d_model=32,
                num_iterations=1,
            )
        except Exception:
            # If an exception occurs, that's expected for this test
            pass


if __name__ == "__main__":
    # Set random seed for reproducible tests
    torch.manual_seed(42)

    # Run tests
    unittest.main(verbosity=2)
