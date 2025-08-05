"""
Unit tests for transformer model components.

Tests cover:
- PositionalEncoding: Sinusoidal encoding generation and application
- FeedForward: Position-wise feed-forward network functionality
- TransformerBlock: Integration of attention and feed-forward layers
- TransformerLM: Full language model functionality, training, generation
"""

import unittest
import torch
import torch.nn as nn
import math

try:
    from src.transformer_model import (
        PositionalEncoding,
        FeedForward,
        TransformerBlock,
        TransformerLM,
        create_causal_mask,
        count_parameters,
        get_model_size_mb,
    )
except ImportError:
    # Fallback for direct test execution
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from transformer_model import (
        PositionalEncoding,
        FeedForward,
        TransformerBlock,
        TransformerLM,
        create_causal_mask,
        count_parameters,
        get_model_size_mb,
    )


class TestPositionalEncoding(unittest.TestCase):
    """Test suite for PositionalEncoding."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.max_len = 1000
        self.pos_enc = PositionalEncoding(self.d_model, self.max_len)

    def test_initialization(self):
        """Test proper initialization of positional encoding."""
        # Check that pe buffer is created
        self.assertTrue(hasattr(self.pos_enc, "pe"))
        self.assertEqual(self.pos_enc.pe.shape, (1, self.max_len, self.d_model))

        # Check that pe is registered as buffer
        self.assertIn("pe", dict(self.pos_enc.named_buffers()))

    def test_sinusoidal_pattern(self):
        """Test that positional encoding follows sinusoidal pattern."""
        pe = self.pos_enc.pe.squeeze(0)  # Remove batch dimension

        # Check that even dimensions use sine, odd dimensions use cosine
        for pos in [0, 1, 10, 100]:
            for i in range(0, min(10, self.d_model), 2):  # Check first few dimensions
                # Even dimensions should use sine
                expected_sine = math.sin(pos / (10000 ** (i / self.d_model)))
                actual_sine = pe[pos, i].item()
                self.assertAlmostEqual(actual_sine, expected_sine, places=5)

                # Odd dimensions should use cosine (if d_model > 1)
                if i + 1 < self.d_model:
                    expected_cosine = math.cos(pos / (10000 ** (i / self.d_model)))
                    actual_cosine = pe[pos, i + 1].item()
                    self.assertAlmostEqual(actual_cosine, expected_cosine, places=5)

    def test_forward_shape(self):
        """Test that forward pass maintains input shape."""
        batch_size, seq_len = 4, 100
        x = torch.randn(batch_size, seq_len, self.d_model)

        output = self.pos_enc(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_forward_adds_positional_info(self):
        """Test that positional encoding is added to input."""
        batch_size, seq_len = 2, 50
        x = torch.zeros(batch_size, seq_len, self.d_model)

        output = self.pos_enc(x)

        # Output should not be zero (positional encoding added)
        self.assertFalse(torch.allclose(output, x))

        # But it should equal the positional encoding (since input was zero)
        expected = self.pos_enc.pe[:, :seq_len]
        self.assertTrue(torch.allclose(output, expected))

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        batch_size = 2
        for seq_len in [1, 10, 50, 100, 500]:
            if seq_len <= self.max_len:
                x = torch.randn(batch_size, seq_len, self.d_model)
                output = self.pos_enc(x)
                self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_odd_d_model(self):
        """Test positional encoding with odd d_model."""
        d_model_odd = 513
        pos_enc_odd = PositionalEncoding(d_model_odd, 100)

        x = torch.randn(2, 50, d_model_odd)
        output = pos_enc_odd(x)

        self.assertEqual(output.shape, (2, 50, d_model_odd))
        self.assertFalse(torch.isnan(output).any())

    def test_edge_case_d_model_1(self):
        """Test positional encoding with d_model=1."""
        pos_enc_1 = PositionalEncoding(1, 100)

        x = torch.randn(2, 50, 1)
        output = pos_enc_1(x)

        self.assertEqual(output.shape, (2, 50, 1))
        self.assertFalse(torch.isnan(output).any())


class TestFeedForward(unittest.TestCase):
    """Test suite for FeedForward layer."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.d_ff = 2048
        self.dropout = 0.1
        self.ff = FeedForward(self.d_model, self.d_ff, self.dropout)

    def test_initialization(self):
        """Test proper initialization of feed-forward layer."""
        self.assertEqual(self.ff.linear1.in_features, self.d_model)
        self.assertEqual(self.ff.linear1.out_features, self.d_ff)
        self.assertEqual(self.ff.linear2.in_features, self.d_ff)
        self.assertEqual(self.ff.linear2.out_features, self.d_model)
        self.assertEqual(self.ff.dropout.p, self.dropout)

    def test_forward_shape(self):
        """Test that forward pass maintains correct shape."""
        batch_size, seq_len = 4, 100
        x = torch.randn(batch_size, seq_len, self.d_model)

        output = self.ff(x)

        self.assertEqual(output.shape, (batch_size, seq_len, self.d_model))

    def test_forward_no_nan(self):
        """Test that forward pass doesn't produce NaN values."""
        x = torch.randn(2, 50, self.d_model)

        self.ff.eval()  # Disable dropout for consistent testing
        output = self.ff(x)

        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_gelu_activation(self):
        """Test that GELU activation is applied correctly."""
        # Create a simple test to verify GELU is used
        x = torch.randn(1, 1, self.d_model)

        # Get intermediate output after first linear + GELU
        with torch.no_grad():
            intermediate = torch.nn.functional.gelu(self.ff.linear1(x))
            expected_output = self.ff.linear2(intermediate)

            self.ff.eval()  # Disable dropout
            actual_output = self.ff(x)

            # Should be close (within numerical precision)
            self.assertTrue(torch.allclose(actual_output, expected_output, atol=1e-6))

    def test_different_dimensions(self):
        """Test with different d_model and d_ff dimensions."""
        test_configs = [(128, 512), (256, 1024), (768, 3072)]

        for d_model, d_ff in test_configs:
            ff = FeedForward(d_model, d_ff, dropout=0.0)
            x = torch.randn(2, 10, d_model)

            output = ff(x)
            self.assertEqual(output.shape, (2, 10, d_model))
            self.assertFalse(torch.isnan(output).any())


class TestTransformerBlock(unittest.TestCase):
    """Test suite for TransformerBlock."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 256
        self.num_heads = 8
        self.d_ff = 1024
        self.seq_len = 64
        self.batch_size = 2

    def test_initialization_standard(self):
        """Test initialization with standard attention."""
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_type="standard",
        )

        self.assertIsNotNone(block.attention)
        self.assertIsNotNone(block.feed_forward)
        self.assertIsNotNone(block.norm1)
        self.assertIsNotNone(block.norm2)

    def test_initialization_sparse(self):
        """Test initialization with sparse attention."""
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_type="sparse",
            sparsity_pattern="local",
            window_size=32,
        )

        self.assertIsNotNone(block.attention)
        # Check that it's the right type of attention
        self.assertEqual(block.attention.sparsity_pattern, "local")

    def test_initialization_flash(self):
        """Test initialization with flash attention."""
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_type="flash",
            block_size=32,
        )

        self.assertIsNotNone(block.attention)
        self.assertEqual(block.attention.block_size, 32)

    def test_invalid_attention_type(self):
        """Test that invalid attention type raises error."""
        with self.assertRaises(ValueError):
            TransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                attention_type="invalid",
            )

    def test_forward_shape(self):
        """Test that forward pass maintains correct shape."""
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_type="standard",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = block(x)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_type="standard",
        )

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mask = create_causal_mask(self.seq_len, x.device)

        output = block(x, mask)

        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        self.assertFalse(torch.isnan(output).any())

    def test_residual_connections(self):
        """Test that residual connections work correctly."""
        block = TransformerBlock(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            attention_type="standard",
            dropout=0.0,  # Disable dropout for this test
        )

        # Use identity initialization to test residual connections
        block.eval()

        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = block(x)

        # Output should be different from input (layers are active)
        self.assertFalse(torch.allclose(output, x))

        # But should not be too different (residual connections help)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())


class TestTransformerLM(unittest.TestCase):
    """Test suite for TransformerLM language model."""

    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.d_model = 256
        self.num_heads = 8
        self.num_layers = 4
        self.d_ff = 1024
        self.max_seq_len = 512
        self.seq_len = 64
        self.batch_size = 2

    def test_initialization(self):
        """Test proper initialization of language model."""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
        )

        self.assertEqual(model.vocab_size, self.vocab_size)
        self.assertEqual(model.d_model, self.d_model)
        self.assertEqual(model.max_seq_len, self.max_seq_len)
        self.assertEqual(len(model.layers), self.num_layers)

        # Check that embedding and output weights are tied
        self.assertTrue(torch.equal(model.token_embedding.weight, model.lm_head.weight))

    def test_forward_inference(self):
        """Test forward pass during inference."""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        outputs = model(input_ids)

        # Check output structure
        self.assertIn("logits", outputs)
        self.assertNotIn("loss", outputs)  # No targets provided

        # Check logits shape
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        self.assertEqual(outputs["logits"].shape, expected_shape)

        # Check that logits are finite
        self.assertTrue(torch.isfinite(outputs["logits"]).all())

    def test_forward_training(self):
        """Test forward pass during training with targets."""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
        )

        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        outputs = model(input_ids, targets=input_ids)

        # Check output structure
        self.assertIn("logits", outputs)
        self.assertIn("loss", outputs)

        # Check loss properties
        loss = outputs["loss"]
        self.assertIsInstance(loss.item(), float)
        self.assertGreater(loss.item(), 0)  # Loss should be positive
        self.assertTrue(torch.isfinite(loss))

    def test_different_attention_types(self):
        """Test model with different attention mechanisms."""
        attention_types = ["standard", "sparse", "flash"]

        for attention_type in attention_types:
            kwargs = {}
            if attention_type == "sparse":
                kwargs = {"sparsity_pattern": "local", "window_size": 32}
            elif attention_type == "flash":
                kwargs = {"block_size": 32}

            model = TransformerLM(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                num_heads=self.num_heads,
                num_layers=2,  # Smaller for testing
                attention_type=attention_type,
                **kwargs,
            )

            input_ids = torch.randint(
                0, self.vocab_size, (self.batch_size, self.seq_len)
            )
            outputs = model(input_ids)

            # Should produce valid outputs regardless of attention type
            self.assertIn("logits", outputs)
            self.assertTrue(torch.isfinite(outputs["logits"]).all())

    def test_generation(self):
        """Test text generation functionality."""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            d_model=128,  # Smaller for faster testing
            num_heads=4,
            num_layers=2,
            max_seq_len=self.max_seq_len,
        )

        # Start with a simple prompt
        input_ids = torch.randint(0, self.vocab_size, (1, 10))

        # Generate some tokens
        generated = model.generate(input_ids, max_new_tokens=20, temperature=1.0)

        # Check that we generated the expected number of tokens
        expected_length = 10 + 20
        self.assertEqual(generated.shape[1], expected_length)

        # Check that generated tokens are valid
        self.assertTrue((generated >= 0).all())
        self.assertTrue((generated < self.vocab_size).all())

    def test_generation_with_sampling_parameters(self):
        """Test generation with different sampling parameters."""
        model = TransformerLM(
            vocab_size=self.vocab_size, d_model=128, num_heads=4, num_layers=2
        )

        input_ids = torch.randint(0, self.vocab_size, (1, 5))

        # Test with top-k sampling
        generated_k = model.generate(input_ids, max_new_tokens=10, top_k=50)
        self.assertEqual(generated_k.shape[1], 15)

        # Test with top-p sampling
        generated_p = model.generate(input_ids, max_new_tokens=10, top_p=0.9)
        self.assertEqual(generated_p.shape[1], 15)

        # Test with temperature
        generated_temp = model.generate(input_ids, max_new_tokens=10, temperature=0.5)
        self.assertEqual(generated_temp.shape[1], 15)

    def test_weight_initialization(self):
        """Test that model weights are initialized properly."""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        )

        # Check that most weights are not all zero (some bias terms might be zero)
        non_zero_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if not torch.allclose(param, torch.zeros_like(param)):
                    non_zero_params += 1

        # Most parameters should be non-zero (allow some bias terms to be zero)
        self.assertGreater(non_zero_params / total_params, 0.5)

        # Check that layer norm weights are initialized correctly
        self.assertTrue(
            torch.allclose(model.norm.weight, torch.ones_like(model.norm.weight))
        )
        self.assertTrue(
            torch.allclose(model.norm.bias, torch.zeros_like(model.norm.bias))
        )

    def test_maximum_sequence_length_handling(self):
        """Test model behavior with sequences at maximum length."""
        model = TransformerLM(
            vocab_size=self.vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,  # Small max length for testing
        )

        # Test with sequence at max length
        input_ids = torch.randint(0, self.vocab_size, (1, 64))
        outputs = model(input_ids)

        self.assertEqual(outputs["logits"].shape[1], 64)
        self.assertTrue(torch.isfinite(outputs["logits"]).all())

    def test_small_vocabulary(self):
        """Test model with very small vocabulary."""
        small_vocab_model = TransformerLM(
            vocab_size=10,  # Very small vocabulary
            d_model=64,
            num_heads=2,
            num_layers=2,
        )

        input_ids = torch.randint(0, 10, (1, 20))
        outputs = small_vocab_model(input_ids)

        self.assertEqual(outputs["logits"].shape[-1], 10)
        self.assertTrue(torch.isfinite(outputs["logits"]).all())


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""

    def test_count_parameters(self):
        """Test parameter counting function."""
        model = TransformerLM(vocab_size=1000, d_model=256, num_heads=8, num_layers=2)

        param_count = count_parameters(model)

        # Should be a positive integer
        self.assertIsInstance(param_count, int)
        self.assertGreater(param_count, 0)

        # Manual verification with a simple model
        simple_model = nn.Linear(10, 5)  # 10*5 + 5 = 55 parameters
        simple_count = count_parameters(simple_model)
        self.assertEqual(simple_count, 55)

    def test_get_model_size_mb(self):
        """Test model size calculation function."""
        model = TransformerLM(vocab_size=1000, d_model=256, num_heads=8, num_layers=2)

        size_mb = get_model_size_mb(model)

        # Should be a positive float
        self.assertIsInstance(size_mb, float)
        self.assertGreater(size_mb, 0)

        # Should be reasonable (not too small or too large)
        self.assertLess(size_mb, 1000)  # Less than 1GB
        self.assertGreater(size_mb, 0.1)  # More than 0.1MB

    def test_create_causal_mask_function(self):
        """Test causal mask creation function."""
        seq_len = 8
        device = torch.device("cpu")

        mask = create_causal_mask(seq_len, device)

        # Check basic properties
        self.assertEqual(mask.shape, (seq_len, seq_len))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.device, device)

        # Check causal structure
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    self.assertTrue(mask[i, j].item())
                else:
                    self.assertFalse(mask[i, j].item())


class TestModelIntegration(unittest.TestCase):
    """Integration tests for model components working together."""

    def test_full_forward_backward_pass(self):
        """Test that the model can perform a complete forward and backward pass."""
        model = TransformerLM(
            vocab_size=100, d_model=64, num_heads=4, num_layers=2, learning_rate=1e-4
        )

        # Create some input data
        input_ids = torch.randint(0, 100, (2, 20))

        # Forward pass
        outputs = model(input_ids, targets=input_ids)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                # Gradients should not be all zero (model is learning something)
                self.assertFalse(
                    torch.allclose(param.grad, torch.zeros_like(param.grad))
                )

    def test_model_in_different_modes(self):
        """Test model behavior in train vs eval mode."""
        model = TransformerLM(
            vocab_size=100, d_model=64, num_heads=4, num_layers=2, dropout=0.1
        )

        input_ids = torch.randint(0, 100, (1, 10))

        # Test in training mode
        model.train()
        train_output = model(input_ids)

        # Test in eval mode
        model.eval()
        eval_output = model(input_ids)

        # Outputs should be different due to dropout
        self.assertFalse(
            torch.allclose(train_output["logits"], eval_output["logits"], atol=1e-6)
        )

        # But both should be valid
        self.assertTrue(torch.isfinite(train_output["logits"]).all())
        self.assertTrue(torch.isfinite(eval_output["logits"]).all())


if __name__ == "__main__":
    # Set random seed for reproducible tests
    torch.manual_seed(42)

    # Run tests
    unittest.main(verbosity=2)
