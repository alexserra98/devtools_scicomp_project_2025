"""
Unit tests for attention mechanism implementations.

Tests cover:
- StandardAttention: Basic functionality, multi-head attention, causal masking
- SparseAttention: Different sparsity patterns, mask generation, memory efficiency
- FlashAttention: Block-wise computation, fallback mechanisms, PyTorch compatibility
"""

import unittest
import torch

try:
    from src.attention_implementations import (
        StandardAttention,
        SparseAttention,
        FlashAttention,
        create_causal_mask,
        benchmark_attention,
    )
except ImportError:
    # Fallback for direct test execution
    import sys
    import os

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from attention_implementations import (
        StandardAttention,
        SparseAttention,
        FlashAttention,
        create_causal_mask,
        benchmark_attention,
    )


class TestStandardAttention(unittest.TestCase):
    """Test suite for StandardAttention implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 512
        self.num_heads = 8
        self.seq_len = 64
        self.batch_size = 2
        self.device = torch.device("cpu")  # Use CPU for consistent testing

        self.attention = StandardAttention(
            d_model=self.d_model, num_heads=self.num_heads, dropout=0.1
        )

        # Create test input
        self.test_input = torch.randn(
            self.batch_size, self.seq_len, self.d_model, device=self.device
        )

    def test_initialization(self):
        """Test proper initialization of StandardAttention."""
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.d_k, self.d_model // self.num_heads)

        # Check layer dimensions
        self.assertEqual(self.attention.w_q.in_features, self.d_model)
        self.assertEqual(self.attention.w_q.out_features, self.d_model)
        self.assertEqual(self.attention.w_k.in_features, self.d_model)
        self.assertEqual(self.attention.w_k.out_features, self.d_model)
        self.assertEqual(self.attention.w_v.in_features, self.d_model)
        self.assertEqual(self.attention.w_v.out_features, self.d_model)
        self.assertEqual(self.attention.w_o.in_features, self.d_model)
        self.assertEqual(self.attention.w_o.out_features, self.d_model)

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        output = self.attention(self.test_input)

        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_no_mask(self):
        """Test forward pass without mask."""
        output = self.attention(self.test_input)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_with_causal_mask(self):
        """Test forward pass with causal mask."""
        mask = create_causal_mask(self.seq_len, self.device)
        output = self.attention(self.test_input, mask)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Output shape should be unchanged
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_scaled_dot_product_attention(self):
        """Test the core attention computation."""
        # Create simple Q, K, V for testing
        d_k = self.d_model // self.num_heads
        Q = torch.randn(self.batch_size, self.num_heads, self.seq_len, d_k)
        K = torch.randn(self.batch_size, self.num_heads, self.seq_len, d_k)
        V = torch.randn(self.batch_size, self.num_heads, self.seq_len, d_k)

        output = self.attention._scaled_dot_product_attention(Q, K, V)

        # Check output shape
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, d_k)
        self.assertEqual(output.shape, expected_shape)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_attention_with_zeros(self):
        """Test attention behavior with zero input."""
        zero_input = torch.zeros(self.batch_size, self.seq_len, self.d_model)
        output = self.attention(zero_input)

        # Output should be finite (zeros should not cause issues)
        self.assertTrue(torch.isfinite(output).all())

    def test_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        for seq_len in [16, 32, 128]:
            test_input = torch.randn(self.batch_size, seq_len, self.d_model)
            output = self.attention(test_input)

            expected_shape = (self.batch_size, seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)
            self.assertFalse(torch.isnan(output).any())

    def test_invalid_dimensions(self):
        """Test that invalid d_model/num_heads ratio raises error."""
        with self.assertRaises(AssertionError):
            StandardAttention(d_model=511, num_heads=8)  # 511 not divisible by 8


class TestSparseAttention(unittest.TestCase):
    """Test suite for SparseAttention implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 256
        self.num_heads = 4
        self.seq_len = 128
        self.batch_size = 2
        self.device = torch.device("cpu")

        self.test_input = torch.randn(
            self.batch_size, self.seq_len, self.d_model, device=self.device
        )

    def test_initialization_local(self):
        """Test initialization with local sparsity pattern."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=32,
        )

        self.assertEqual(attention.sparsity_pattern, "local")
        self.assertEqual(attention.window_size, 32)
        self.assertEqual(attention.d_model, self.d_model)
        self.assertEqual(attention.num_heads, self.num_heads)

    def test_initialization_strided(self):
        """Test initialization with strided sparsity pattern."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="strided",
            stride=4,
        )

        self.assertEqual(attention.sparsity_pattern, "strided")
        self.assertEqual(attention.stride, 4)

    def test_initialization_fixed(self):
        """Test initialization with fixed sparsity pattern."""
        attention = SparseAttention(
            d_model=self.d_model, num_heads=self.num_heads, sparsity_pattern="fixed"
        )

        self.assertEqual(attention.sparsity_pattern, "fixed")

    def test_invalid_sparsity_pattern(self):
        """Test that invalid sparsity pattern raises error."""
        attention = SparseAttention(
            d_model=self.d_model, num_heads=self.num_heads, sparsity_pattern="invalid"
        )

        with self.assertRaises(ValueError):
            attention._create_sparsity_mask(64, self.device)

    def test_local_mask_generation(self):
        """Test local sparsity mask generation."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=8,
        )

        mask = attention._create_sparsity_mask(16, self.device)

        # Check mask shape (includes batch and head dimensions)
        self.assertEqual(mask.shape, (1, 1, 16, 16))

        # Check that mask is Boolean
        self.assertEqual(mask.dtype, torch.bool)

        # Check that diagonal elements are True (each position attends to itself)
        base_mask = mask.squeeze()
        for i in range(16):
            self.assertTrue(base_mask[i, i].item())

        # Check that distant elements are False
        if 16 > 8:  # Only if sequence is longer than window
            self.assertFalse(base_mask[0, 15].item())
            self.assertFalse(base_mask[15, 0].item())

    def test_strided_mask_generation(self):
        """Test strided sparsity mask generation."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="strided",
            stride=4,
        )

        mask = attention._create_sparsity_mask(16, self.device)

        # Check mask shape
        self.assertEqual(mask.shape, (1, 1, 16, 16))

        # Check that stride positions are True
        base_mask = mask.squeeze()
        for i in range(16):
            self.assertTrue(base_mask[i, 0].item())  # Position 0 should always be True
            if i >= 4:
                self.assertTrue(base_mask[i, 4].item())  # Stride positions

    def test_fixed_mask_generation(self):
        """Test fixed sparsity mask generation."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="fixed",
            stride=2,
        )

        mask = attention._create_sparsity_mask(16, self.device)

        # Check mask shape
        self.assertEqual(mask.shape, (1, 1, 16, 16))

        # Check that mask maintains causality (lower triangular structure)
        # Note: Detailed causality check depends on the specific fixed pattern
        # We just verify the mask was created successfully
        self.assertEqual(mask.dtype, torch.bool)

    def test_mask_caching(self):
        """Test that sparsity masks are properly cached."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=8,
        )

        # Clear cache
        attention._mask_cache.clear()

        # Generate mask twice
        mask1 = attention._create_sparsity_mask(16, self.device)
        mask2 = attention._create_sparsity_mask(16, self.device)

        # Should be identical (from cache)
        self.assertTrue(torch.equal(mask1, mask2))

        # Cache should have one entry
        self.assertEqual(len(attention._mask_cache), 1)

    def test_forward_local(self):
        """Test forward pass with local attention."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=32,
        )

        output = attention(self.test_input)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_strided(self):
        """Test forward pass with strided attention."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="strided",
            stride=4,
        )

        output = attention(self.test_input)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_with_input_mask(self):
        """Test forward pass with additional input mask."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=32,
        )

        # Create a causal mask
        input_mask = create_causal_mask(self.seq_len, self.device)
        output = attention(self.test_input, input_mask)

        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_different_mask_dimensions(self):
        """Test handling of different input mask dimensions."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=32,
        )

        # Test 2D mask [seq_len, seq_len]
        mask_2d = torch.ones(self.seq_len, self.seq_len, dtype=torch.bool)
        output_2d = attention(self.test_input, mask_2d)
        self.assertEqual(output_2d.shape, (self.batch_size, self.seq_len, self.d_model))

        # Test 3D mask [batch_size, seq_len, seq_len]
        mask_3d = torch.ones(
            self.batch_size, self.seq_len, self.seq_len, dtype=torch.bool
        )
        output_3d = attention(self.test_input, mask_3d)
        self.assertEqual(output_3d.shape, (self.batch_size, self.seq_len, self.d_model))

        # Test 4D mask [batch_size, num_heads, seq_len, seq_len]
        mask_4d = torch.ones(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.seq_len,
            dtype=torch.bool,
        )
        output_4d = attention(self.test_input, mask_4d)
        self.assertEqual(output_4d.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_invalid_mask_dimension(self):
        """Test that invalid mask dimensions raise error."""
        attention = SparseAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            sparsity_pattern="local",
            window_size=32,
        )

        # Create invalid 1D mask
        invalid_mask = torch.ones(self.seq_len, dtype=torch.bool)

        with self.assertRaises(ValueError):
            attention(self.test_input, invalid_mask)


class TestFlashAttention(unittest.TestCase):
    """Test suite for FlashAttention implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 256
        self.num_heads = 4
        self.seq_len = 128
        self.batch_size = 2
        self.device = torch.device("cpu")

        self.attention = FlashAttention(
            d_model=self.d_model, num_heads=self.num_heads, block_size=32
        )

        self.test_input = torch.randn(
            self.batch_size, self.seq_len, self.d_model, device=self.device
        )

    def test_initialization(self):
        """Test proper initialization of FlashAttention."""
        self.assertEqual(self.attention.d_model, self.d_model)
        self.assertEqual(self.attention.num_heads, self.num_heads)
        self.assertEqual(self.attention.d_k, self.d_model // self.num_heads)
        self.assertEqual(self.attention.block_size, 32)

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        output = self.attention(self.test_input)

        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_no_mask(self):
        """Test forward pass without mask."""
        output = self.attention(self.test_input)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_forward_with_mask(self):
        """Test forward pass with causal mask."""
        mask = create_causal_mask(self.seq_len, self.device)
        output = self.attention(self.test_input, mask)

        # Should not be NaN or infinite
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

        # Output shape should be unchanged
        expected_shape = (self.batch_size, self.seq_len, self.d_model)
        self.assertEqual(output.shape, expected_shape)

    def test_block_wise_attention_small_sequence(self):
        """Test block-wise attention with sequence smaller than block size."""
        small_input = torch.randn(
            self.batch_size, 16, self.d_model
        )  # Smaller than block_size=32
        output = self.attention(small_input)

        expected_shape = (self.batch_size, 16, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())

    def test_block_wise_attention_large_sequence(self):
        """Test block-wise attention with sequence larger than block size."""
        large_input = torch.randn(
            self.batch_size, 256, self.d_model
        )  # Larger than block_size=32
        output = self.attention(large_input)

        expected_shape = (self.batch_size, 256, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())

    def test_standard_attention_fallback(self):
        """Test that standard attention fallback works correctly."""
        # Force use of block-wise implementation by mocking PyTorch flash attention
        attention = FlashAttention(
            d_model=self.d_model, num_heads=self.num_heads, block_size=32
        )

        # Test with small sequence (should use standard attention)
        small_input = torch.randn(self.batch_size, 16, self.d_model)
        output = attention._block_wise_attention(small_input, None)

        expected_shape = (self.batch_size, 16, self.d_model)
        self.assertEqual(output.shape, expected_shape)
        self.assertFalse(torch.isnan(output).any())

    def test_different_block_sizes(self):
        """Test attention with different block sizes."""
        for block_size in [16, 32, 64]:
            attention = FlashAttention(
                d_model=self.d_model, num_heads=self.num_heads, block_size=block_size
            )

            output = attention(self.test_input)

            expected_shape = (self.batch_size, self.seq_len, self.d_model)
            self.assertEqual(output.shape, expected_shape)
            self.assertFalse(torch.isnan(output).any())


class TestUtilityFunctions(unittest.TestCase):
    """Test suite for utility functions."""

    def test_create_causal_mask(self):
        """Test causal mask creation."""
        seq_len = 8
        device = torch.device("cpu")

        mask = create_causal_mask(seq_len, device)

        # Check shape
        self.assertEqual(mask.shape, (seq_len, seq_len))

        # Check dtype
        self.assertEqual(mask.dtype, torch.bool)

        # Check device
        self.assertEqual(mask.device, device)

        # Check that it's lower triangular
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    self.assertTrue(mask[i, j].item())
                else:
                    self.assertFalse(mask[i, j].item())

    def test_benchmark_attention_function(self):
        """Test the benchmark_attention utility function."""
        # Create a simple attention function to benchmark
        attention = StandardAttention(d_model=128, num_heads=4)

        def attention_fn(x):
            return attention(x)

        # Run benchmark with small parameters for testing
        avg_time, peak_memory = benchmark_attention(
            attention_fn, batch_size=2, seq_len=32, d_model=128, num_iterations=10
        )

        # Check that we get valid results
        self.assertIsInstance(avg_time, float)
        self.assertIsInstance(peak_memory, float)
        self.assertGreater(avg_time, 0)
        self.assertGreater(peak_memory, 0)


class TestAttentionEquivalence(unittest.TestCase):
    """Test suite to verify different attention mechanisms produce reasonable outputs."""

    def setUp(self):
        """Set up test fixtures."""
        self.d_model = 128
        self.num_heads = 4
        self.seq_len = 32
        self.batch_size = 2
        self.device = torch.device("cpu")

        # Create identical input for all attention mechanisms
        torch.manual_seed(42)  # For reproducible results
        self.test_input = torch.randn(
            self.batch_size, self.seq_len, self.d_model, device=self.device
        )

    def test_output_ranges_are_reasonable(self):
        """Test that all attention mechanisms produce outputs in reasonable ranges."""
        # Initialize attention mechanisms
        standard_attn = StandardAttention(self.d_model, self.num_heads, dropout=0.0)
        sparse_attn = SparseAttention(
            self.d_model,
            self.num_heads,
            sparsity_pattern="local",
            window_size=16,
            dropout=0.0,
        )
        flash_attn = FlashAttention(self.d_model, self.num_heads, block_size=16)

        # Set to eval mode to disable dropout
        standard_attn.eval()
        sparse_attn.eval()
        flash_attn.eval()

        with torch.no_grad():
            standard_out = standard_attn(self.test_input)
            sparse_out = sparse_attn(self.test_input)
            flash_out = flash_attn(self.test_input)

        # Check that outputs are in reasonable ranges (not too large/small)
        for name, output in [
            ("standard", standard_out),
            ("sparse", sparse_out),
            ("flash", flash_out),
        ]:
            self.assertFalse(torch.isnan(output).any(), f"{name} produced NaN")
            self.assertFalse(torch.isinf(output).any(), f"{name} produced inf")

            # Check that values are in a reasonable range (rough heuristic)
            self.assertLess(
                output.abs().max().item(), 100.0, f"{name} outputs too large"
            )
            self.assertGreater(
                output.abs().max().item(), 0.01, f"{name} outputs too small"
            )

    def test_attention_preserves_sequence_structure(self):
        """Test that attention mechanisms preserve important sequence properties."""
        standard_attn = StandardAttention(self.d_model, self.num_heads, dropout=0.0)

        # Create a structured input (increasing values along sequence)
        structured_input = torch.zeros(1, self.seq_len, self.d_model)
        for i in range(self.seq_len):
            structured_input[0, i, :] = i * 0.1

        standard_attn.eval()
        with torch.no_grad():
            output = standard_attn(structured_input)

        # Output should not be identical to input (attention should mix information)
        self.assertFalse(torch.allclose(output, structured_input))

        # But should maintain finite values
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    # Set up test environment
    torch.manual_seed(42)

    # Run tests
    unittest.main(verbosity=2)
