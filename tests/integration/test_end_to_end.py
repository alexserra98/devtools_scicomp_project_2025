"""
Integration tests for the transformer project.

Tests cover:
- End-to-end model training workflows
- Attention mechanism interoperability
- Benchmarking pipeline integration
- Model generation and evaluation pipelines
"""

import unittest
import torch
import tempfile
import os
from torch.utils.data import DataLoader

try:
    from src.transformer_model import TransformerLM, TransformerBlock
    from src.benchmark import SyntheticTextDataset, PerformanceBenchmark
except ImportError:
    # Fallback for direct test execution
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from transformer_model import TransformerLM, TransformerBlock
    from benchmark import SyntheticTextDataset, PerformanceBenchmark


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete workflows from data to results."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        torch.manual_seed(42)  # For reproducible tests

    def test_complete_training_workflow(self):
        """Test complete model training workflow."""
        # Create model
        model = TransformerLM(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            attention_type="standard",
            learning_rate=1e-3,
        )

        # Create dataset
        dataset = SyntheticTextDataset(vocab_size=100, seq_len=32, num_samples=50)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Training loop
        model.train()
        initial_loss = None
        final_loss = None

        for epoch in range(3):
            epoch_losses = []
            for batch in dataloader:
                input_ids = batch["input_ids"]

                optimizer.zero_grad()
                outputs = model(input_ids, targets=input_ids)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            if initial_loss is None:
                initial_loss = avg_loss
            final_loss = avg_loss

        # Check that training progressed
        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertGreater(initial_loss, 0)
        self.assertGreater(final_loss, 0)

        # Loss should generally decrease (though not guaranteed in few epochs)
        # We just check that final loss is reasonable
        self.assertLess(final_loss, 10.0)  # Reasonable for random data

    def test_generation_workflow(self):
        """Test text generation workflow."""
        # Create and train a small model
        model = TransformerLM(
            vocab_size=50, d_model=32, num_heads=2, num_layers=2, max_seq_len=64
        )

        # Quick training on small data
        dataset = SyntheticTextDataset(vocab_size=50, seq_len=16, num_samples=20)
        dataloader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"]
            optimizer.zero_grad()
            outputs = model(input_ids, targets=input_ids)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        # Test generation
        model.eval()
        prompt = torch.randint(0, 50, (1, 5))

        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)

        # Check generation results
        self.assertEqual(generated.shape[0], 1)  # Batch size
        self.assertEqual(generated.shape[1], 15)  # Original + new tokens
        self.assertTrue((generated >= 0).all())
        self.assertTrue((generated < 50).all())

    def test_evaluation_workflow(self):
        """Test model evaluation workflow."""
        # Create models with different attention types
        models = {}
        for attention_type in ["standard", "sparse"]:
            kwargs = {}
            if attention_type == "sparse":
                kwargs = {"sparsity_pattern": "local", "window_size": 16}

            model = TransformerLM(
                vocab_size=100,
                d_model=64,
                num_heads=4,
                num_layers=2,
                attention_type=attention_type,
                **kwargs,
            )
            models[attention_type] = model

        # Create evaluation data
        eval_dataset = SyntheticTextDataset(vocab_size=100, seq_len=32, num_samples=20)
        eval_dataloader = DataLoader(eval_dataset, batch_size=4)

        # Evaluate models
        results = {}
        for name, model in models.items():
            model.eval()
            total_loss = 0
            total_batches = 0

            with torch.no_grad():
                for batch in eval_dataloader:
                    input_ids = batch["input_ids"]
                    outputs = model(input_ids, targets=input_ids)
                    total_loss += outputs["loss"].item()
                    total_batches += 1

            avg_loss = total_loss / total_batches
            results[name] = avg_loss

        # Check that all models produced reasonable results
        for name, loss in results.items():
            self.assertGreater(loss, 0)
            self.assertLess(loss, 20)  # Reasonable for untrained models
            self.assertFalse(torch.isnan(torch.tensor(loss)))


class TestAttentionInteroperability(unittest.TestCase):
    """Test that different attention mechanisms work interchangeably."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 128
        self.num_heads = 4
        self.seq_len = 32
        self.batch_size = 2

        self.test_input = torch.randn(self.batch_size, self.seq_len, self.d_model)

    def test_attention_mechanism_swap(self):
        """Test swapping attention mechanisms in transformer blocks."""
        # Test configurations
        attention_configs = [
            ("standard", {}),
            ("sparse", {"sparsity_pattern": "local", "window_size": 16}),
            ("flash", {"block_size": 16}),
        ]

        outputs = {}

        for attention_type, kwargs in attention_configs:
            block = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=512,
                attention_type=attention_type,
                dropout=0.0,  # Disable dropout for consistent testing
                **kwargs,
            )

            block.eval()
            with torch.no_grad():
                output = block(self.test_input)

            outputs[attention_type] = output

            # Check output properties
            self.assertEqual(output.shape, self.test_input.shape)
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

        # All attention mechanisms should produce different but valid outputs
        attention_types = list(outputs.keys())
        for i in range(len(attention_types)):
            for j in range(i + 1, len(attention_types)):
                type1, type2 = attention_types[i], attention_types[j]
                # Outputs should be different (different attention patterns)
                self.assertFalse(
                    torch.allclose(outputs[type1], outputs[type2], atol=1e-6)
                )

    def test_model_with_mixed_attention_types(self):
        """Test models where different layers use different attention types."""
        # This test simulates a model with mixed attention mechanisms
        # (though our current implementation doesn't support this directly,
        # we test the concept)

        layers = []
        attention_types = ["standard", "sparse", "standard"]

        for i, attention_type in enumerate(attention_types):
            kwargs = {}
            if attention_type == "sparse":
                kwargs = {"sparsity_pattern": "local", "window_size": 16}

            layer = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=512,
                attention_type=attention_type,
                **kwargs,
            )
            layers.append(layer)

        # Apply layers sequentially
        x = self.test_input
        for layer in layers:
            layer.eval()
            with torch.no_grad():
                x = layer(x)

        # Final output should be valid
        self.assertEqual(x.shape, self.test_input.shape)
        self.assertFalse(torch.isnan(x).any())
        self.assertFalse(torch.isinf(x).any())

    def test_attention_gradient_compatibility(self):
        """Test that all attention types support gradient computation."""
        attention_configs = [
            ("standard", {}),
            ("sparse", {"sparsity_pattern": "local", "window_size": 16}),
            ("flash", {"block_size": 16}),
        ]

        for attention_type, kwargs in attention_configs:
            block = TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=512,
                attention_type=attention_type,
                **kwargs,
            )

            # Test gradient computation
            input_tensor = self.test_input.clone().requires_grad_(True)
            output = block(input_tensor)

            # Compute loss and backpropagate
            loss = output.sum()
            loss.backward()

            # Check that gradients are computed
            self.assertIsNotNone(input_tensor.grad)
            self.assertFalse(torch.isnan(input_tensor.grad).any())

            # Check that model parameters have gradients
            for param in block.parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad)


class TestBenchmarkIntegration(unittest.TestCase):
    """Test integration of benchmarking components."""

    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = PerformanceBenchmark(device="cpu")

    def test_benchmark_model_lifecycle(self):
        """Test benchmarking throughout model lifecycle."""
        # Create a model for benchmarking
        model = TransformerLM(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            attention_type="standard",
        )

        # Benchmark initial inference
        input_ids = torch.randint(0, 100, (2, 16))

        model.eval()
        with torch.no_grad():
            initial_output = model(input_ids)

        # Quick training
        dataset = SyntheticTextDataset(vocab_size=100, seq_len=16, num_samples=20)
        dataloader = DataLoader(dataset, batch_size=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        model.train()
        for batch in dataloader:
            batch_input = batch["input_ids"]
            optimizer.zero_grad()
            outputs = model(batch_input, targets=batch_input)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        # Benchmark post-training inference
        model.eval()
        with torch.no_grad():
            final_output = model(input_ids)

        # Check that outputs are different (model learned something)
        self.assertFalse(
            torch.allclose(initial_output["logits"], final_output["logits"], atol=1e-4)
        )

        # Both outputs should be valid
        self.assertTrue(torch.isfinite(initial_output["logits"]).all())
        self.assertTrue(torch.isfinite(final_output["logits"]).all())

    def test_attention_benchmark_consistency(self):
        """Test that attention benchmarks produce consistent results."""
        # Run benchmark twice with same parameters
        config = {
            "seq_lengths": [16, 32],
            "d_model": 64,
            "num_heads": 4,
            "batch_size": 2,
            "num_iterations": 3,
        }

        results1 = self.benchmark.benchmark_attention_mechanisms(**config)
        results2 = self.benchmark.benchmark_attention_mechanisms(**config)

        # Results should have similar structure
        self.assertEqual(set(results1.keys()), set(results2.keys()))

        # Runtimes should be in similar ballpark (within factor of 3)
        for attention_type in results1:
            if (
                results1[attention_type]["success"][0]
                and results2[attention_type]["success"][0]
            ):
                runtime1 = results1[attention_type]["runtime"][0]
                runtime2 = results2[attention_type]["runtime"][0]

                # Check that runtimes are reasonable and not too different
                self.assertGreater(runtime1, 0)
                self.assertGreater(runtime2, 0)
                ratio = max(runtime1, runtime2) / min(runtime1, runtime2)
                self.assertLess(
                    ratio, 10.0
                )  # Within 10x (accounting for system variance)

    def test_benchmark_error_recovery(self):
        """Test that benchmarks handle errors gracefully."""
        # Test with parameters that might cause issues
        problematic_config = {
            "seq_lengths": [8],  # Very small
            "d_model": 8,  # Very small
            "num_heads": 2,
            "batch_size": 1,
            "num_iterations": 1,
        }

        # This should not crash even with unusual parameters
        results = self.benchmark.benchmark_attention_mechanisms(**problematic_config)

        # Should get some results
        self.assertIn("standard", results)

        # Results should have correct structure even if some failed
        for attention_type, metrics in results.items():
            self.assertIn("runtime", metrics)
            self.assertIn("memory", metrics)
            self.assertIn("success", metrics)
            self.assertEqual(len(metrics["runtime"]), 1)


class TestModelPersistence(unittest.TestCase):
    """Test model saving/loading and persistence."""

    def test_model_save_load_cycle(self):
        """Test saving and loading trained models."""
        # Create and train a model
        original_model = TransformerLM(
            vocab_size=50, d_model=32, num_heads=2, num_layers=2
        )

        # Quick training
        dataset = SyntheticTextDataset(vocab_size=50, seq_len=16, num_samples=10)
        dataloader = DataLoader(dataset, batch_size=2)
        optimizer = torch.optim.AdamW(original_model.parameters(), lr=1e-3)

        original_model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"]
            optimizer.zero_grad()
            outputs = original_model(input_ids, targets=input_ids)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            model_path = f.name

        try:
            torch.save(original_model.state_dict(), model_path)

            # Create new model and load weights
            loaded_model = TransformerLM(
                vocab_size=50, d_model=32, num_heads=2, num_layers=2
            )
            loaded_model.load_state_dict(torch.load(model_path, map_location="cpu"))

            # Test that models produce identical outputs
            test_input = torch.randint(0, 50, (1, 10))

            original_model.eval()
            loaded_model.eval()

            with torch.no_grad():
                original_output = original_model(test_input)
                loaded_output = loaded_model(test_input)

            # Outputs should be identical
            self.assertTrue(
                torch.allclose(
                    original_output["logits"], loaded_output["logits"], atol=1e-6
                )
            )

        finally:
            os.unlink(model_path)

    def test_model_configuration_persistence(self):
        """Test that model configurations are preserved."""
        configs = [
            {
                "vocab_size": 100,
                "d_model": 64,
                "num_heads": 4,
                "num_layers": 2,
                "attention_type": "standard",
            },
            {
                "vocab_size": 200,
                "d_model": 128,
                "num_heads": 8,
                "num_layers": 3,
                "attention_type": "sparse",
                "sparsity_pattern": "local",
                "window_size": 32,
            },
        ]

        for config in configs:
            model = TransformerLM(**config)

            # Check that configuration is preserved in hyperparameters
            if hasattr(model, "hparams"):
                for key, value in config.items():
                    if key in model.hparams:
                        self.assertEqual(model.hparams[key], value)

            # Check that model properties match configuration
            self.assertEqual(model.vocab_size, config["vocab_size"])
            self.assertEqual(model.d_model, config["d_model"])
            self.assertEqual(len(model.layers), config["num_layers"])


if __name__ == "__main__":
    # Set random seed for reproducible tests
    torch.manual_seed(42)

    # Run tests
    unittest.main(verbosity=2)
