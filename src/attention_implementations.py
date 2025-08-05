import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StandardAttention(nn.Module):
    """
    Standard scaled dot-product attention with optional optimizations.

    Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V

    where Q, K, V are query, key, value matrices and d_k is the key dimension.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)
        V = self.w_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output = self._scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        return self.w_o(attn_output)

    def _scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Implements the core attention computation:
        Attention(Q, K, V) = softmax(QK^T / √d_k)V
        """
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)


class SparseAttention(nn.Module):
    """
    Sparse Attention implementation with configurable sparsity patterns.

    Mathematical formulation:
    Attention(Q, K, V) = softmax(mask ⊙ (QK^T / √d_k))V

    where ⊙ denotes element-wise multiplication and mask defines the sparsity pattern.

    Supported patterns:
    - 'local': Local sliding window attention
    - 'strided': Strided attention pattern
    - 'fixed': Fixed sparsity pattern
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        sparsity_pattern: str = "local",
        window_size: int = 64,
        stride: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sparsity_pattern = sparsity_pattern
        self.window_size = window_size
        self.stride = stride

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Cache for sparsity masks to avoid recomputation
        self._mask_cache = {}

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Generate sparsity mask
        sparsity_mask = self._create_sparsity_mask(seq_len, Q.device)

        # Expand sparsity mask to match Q, K dimensions: [batch_size, num_heads, seq_len, seq_len]
        sparsity_mask = sparsity_mask.expand(
            batch_size, self.num_heads, seq_len, seq_len
        )

        # Combine with input mask if provided
        if mask is not None:
            # Input mask can be [seq_len, seq_len], [batch_size, seq_len, seq_len], or [batch_size, num_heads, seq_len, seq_len]
            if mask.dim() == 2:  # [seq_len, seq_len]
                mask = (
                    mask.unsqueeze(0)
                    .unsqueeze(0)
                    .expand(batch_size, self.num_heads, seq_len, seq_len)
                )
            elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                mask = mask.unsqueeze(1).expand(
                    batch_size, self.num_heads, seq_len, seq_len
                )
            elif mask.dim() == 4:  # Already [batch_size, num_heads, seq_len, seq_len]
                pass
            else:
                raise ValueError(f"Unsupported mask dimension: {mask.dim()}")
            sparsity_mask = sparsity_mask & mask

        # Apply sparse attention
        attn_output = self._sparse_attention(Q, K, V, sparsity_mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        return self.w_o(attn_output)

    def _create_sparsity_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparsity mask based on the specified pattern with caching."""
        cache_key = (
            seq_len,
            device,
            self.sparsity_pattern,
            self.window_size,
            self.stride,
        )

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        if self.sparsity_pattern == "local":
            base_mask = self._local_mask_vectorized(seq_len, device)
        elif self.sparsity_pattern == "strided":
            base_mask = self._strided_mask_vectorized(seq_len, device)
        elif self.sparsity_pattern == "fixed":
            base_mask = self._fixed_mask_vectorized(seq_len, device)
        else:
            raise ValueError(f"Unknown sparsity pattern: {self.sparsity_pattern}")

        # Expand to match (1, 1, seq_len, seq_len) for broadcasting
        mask = base_mask.unsqueeze(0).unsqueeze(0)
        self._mask_cache[cache_key] = mask
        return mask

    def _local_mask_vectorized(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Vectorized local sliding window attention mask."""
        # Create row and column indices
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

        # Calculate distance between positions
        distance = torch.abs(row_idx - col_idx)

        # Create mask for local window
        mask = distance <= (self.window_size // 2)

        return mask

    def _strided_mask_vectorized(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Vectorized strided attention mask - optimized version."""
        # Create indices
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]

        # Local window mask (for efficiency, use smaller window for strided)
        local_window = min(
            self.window_size // 2, 32
        )  # Smaller local window for strided
        distance = torch.abs(row_idx - col_idx)
        local_mask = distance <= local_window

        # Strided mask - only attend to every stride-th position
        # Handle edge case where stride might be 0
        if self.stride > 0:
            strided_mask = (col_idx % self.stride) == 0
        else:
            # If stride is 0, fall back to local mask only
            strided_mask = torch.zeros_like(local_mask, dtype=torch.bool)

        # Combine: local OR strided pattern
        mask = local_mask | strided_mask

        return mask

    def _fixed_mask_vectorized(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Vectorized fixed sparsity pattern (optimized lower triangular)."""
        # Create lower triangular mask
        row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
        col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = row_idx >= col_idx

        # Apply sparsity pattern - keep every stride-th element in some rows
        # Handle edge case where stride might be 0
        if self.stride > 0:
            for i in range(self.stride, seq_len, self.stride * 2):
                # Reset the row and apply stride pattern
                mask[i, :] = False
                mask[i, :: self.stride] = True
                # Ensure causality is maintained
                mask[i, i + 1 :] = False

        return mask

    def _sparse_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Optimized sparse attention computation with actual sparsity benefits."""
        batch_size, num_heads, seq_len, d_k = Q.shape

        # For local patterns, we can optimize by only computing relevant blocks
        if self.sparsity_pattern == "local" and seq_len > self.window_size * 2:
            return self._local_block_attention(Q, K, V, mask)
        elif self.sparsity_pattern == "strided" and seq_len > 128:
            return self._strided_block_attention(Q, K, V, mask)
        else:
            # Fallback to masked attention for other patterns or small sequences
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Apply sparsity mask
            scores = scores.masked_fill(~mask, -float("inf"))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            return torch.matmul(attn_weights, V)

    def _strided_block_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Optimized strided attention using sparse computation."""
        batch_size, num_heads, seq_len, d_k = Q.shape

        # Initialize output tensor
        output = torch.zeros_like(Q)

        # Process in chunks to take advantage of strided pattern
        chunk_size = max(32, self.stride * 4)

        for i in range(0, seq_len, chunk_size):
            i_end = min(i + chunk_size, seq_len)

            # For each position in the chunk, compute attention efficiently
            for qi in range(i, i_end):
                # Determine attention range for this position
                # Local window
                local_start = max(0, qi - 16)  # Smaller local window for strided
                local_end = min(seq_len, qi + 17)

                # Strided positions
                strided_positions = list(range(0, seq_len, self.stride))

                # Combine local and strided positions, remove duplicates
                all_positions = list(
                    set(range(local_start, local_end)) | set(strided_positions)
                )
                all_positions = [p for p in all_positions if p < seq_len]
                all_positions.sort()

                if len(all_positions) == 0:
                    continue

                # Extract relevant K, V
                k_relevant = K[:, :, all_positions, :]  # [batch, heads, relevant, d_k]
                v_relevant = V[:, :, all_positions, :]  # [batch, heads, relevant, d_k]
                q_i = Q[:, :, qi : qi + 1, :]  # [batch, heads, 1, d_k]

                # Compute attention scores
                scores = torch.matmul(q_i, k_relevant.transpose(-2, -1)) / math.sqrt(
                    self.d_k
                )

                # Apply softmax
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                # Compute output
                output[:, :, qi : qi + 1, :] = torch.matmul(attn_weights, v_relevant)

        return output

    def _local_block_attention(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Optimized local attention using block-wise computation."""
        batch_size, num_heads, seq_len, d_k = Q.shape
        half_window = self.window_size // 2

        # Initialize output tensor
        output = torch.zeros_like(Q)

        for i in range(seq_len):
            # Calculate the local window for position i
            k_start = max(0, i - half_window)
            k_end = min(seq_len, i + half_window + 1)

            # Extract relevant Q, K, V
            q_i = Q[:, :, i : i + 1, :]  # [batch, heads, 1, d_k]
            k_local = K[:, :, k_start:k_end, :]  # [batch, heads, window, d_k]
            v_local = V[:, :, k_start:k_end, :]  # [batch, heads, window, d_k]

            # Compute attention scores
            scores = torch.matmul(q_i, k_local.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Compute output
            output[:, :, i : i + 1, :] = torch.matmul(attn_weights, v_local)

        return output


class FlashAttention(nn.Module):
    """
    Flash Attention implementation for memory-efficient attention computation.

    Mathematical formulation:
    Uses block-wise computation to reduce memory complexity from O(n²) to O(n).

    The algorithm computes attention in blocks:
    1. Divide Q, K, V into blocks
    2. Compute attention for each block pair
    3. Use online softmax to maintain numerical stability

    Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
    """

    def __init__(
        self, d_model: int, num_heads: int, block_size: int = 64, dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.block_size = block_size

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        # Use PyTorch's optimized implementation if available
        if hasattr(F, "scaled_dot_product_attention") and torch.cuda.is_available():
            return self._pytorch_flash_attention(x, mask)
        else:
            # Fallback to block-wise implementation
            return self._block_wise_attention(x, mask)

    def _pytorch_flash_attention(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Use PyTorch's optimized flash attention implementation."""
        batch_size, seq_len, d_model = x.size()

        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Use PyTorch's flash attention with optimizations
        if torch.cuda.is_available():
            try:
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True, enable_math=False, enable_mem_efficient=False
                ):
                    attn_output = F.scaled_dot_product_attention(
                        Q,
                        K,
                        V,
                        attn_mask=mask,
                        dropout_p=self.dropout.p if self.training else 0.0,
                    )
            except Exception:
                # Fallback if SDPA with flash fails
                attn_output = F.scaled_dot_product_attention(
                    Q,
                    K,
                    V,
                    attn_mask=mask,
                    dropout_p=self.dropout.p if self.training else 0.0,
                )
        else:
            attn_output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )

        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        return self.w_o(attn_output)

    def _block_wise_attention(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Implement block-wise flash attention algorithm."""
        batch_size, seq_len, d_model = x.size()

        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # If sequence is small, use standard attention
        if seq_len <= self.block_size:
            return self._standard_attention(Q, K, V, mask)

        # Block-wise computation
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        output = torch.zeros_like(Q)

        for i in range(num_blocks):
            q_start = i * self.block_size
            q_end = min((i + 1) * self.block_size, seq_len)
            Q_block = Q[:, :, q_start:q_end, :]  # [batch, heads, block_size, d_k]

            # Initialize output accumulators for this block
            O_block = torch.zeros(
                batch_size,
                self.num_heads,
                q_end - q_start,
                self.d_k,
                device=Q.device,
                dtype=Q.dtype,
            )
            l_block = torch.zeros(
                batch_size,
                self.num_heads,
                q_end - q_start,
                1,
                device=Q.device,
                dtype=Q.dtype,
            )
            m_block = torch.full(
                (batch_size, self.num_heads, q_end - q_start, 1),
                -float("inf"),
                device=Q.device,
                dtype=Q.dtype,
            )

            for j in range(num_blocks):
                k_start = j * self.block_size
                k_end = min((j + 1) * self.block_size, seq_len)
                K_block = K[:, :, k_start:k_end, :]
                V_block = V[:, :, k_start:k_end, :]

                # Compute attention scores for this block pair
                S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(
                    self.d_k
                )

                # Apply mask if provided
                if mask is not None:
                    mask_block = mask[q_start:q_end, k_start:k_end]
                    S_block = S_block.masked_fill(~mask_block, -float("inf"))

                # Online softmax computation
                m_new = torch.maximum(m_block, S_block.max(dim=-1, keepdim=True)[0])

                # Update accumulators
                alpha = torch.exp(m_block - m_new)
                beta = torch.exp(S_block - m_new)

                l_new = alpha * l_block + beta.sum(dim=-1, keepdim=True)

                O_new = (
                    alpha * l_block * O_block + torch.matmul(beta, V_block)
                ) / l_new

                # Update state
                O_block = O_new
                l_block = l_new
                m_block = m_new

            output[:, :, q_start:q_end, :] = O_block

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(output)

    def _standard_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard attention for small sequences."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, -float("inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        batch_size, seq_len = attn_output.shape[0], attn_output.shape[2]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )
        return self.w_o(attn_output)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower triangular) mask for autoregressive models."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


def benchmark_attention(
    attention_fn, batch_size: int, seq_len: int, d_model: int, num_iterations: int = 100
) -> Tuple[float, float]:
    """
    Benchmark attention implementation for runtime and memory usage.

    Returns:
        Tuple of (average_time_ms, peak_memory_mb)
    """
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, d_model, device=device)

    # Warmup
    for _ in range(10):
        _ = attention_fn(x)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        _ = attention_fn(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms

    peak_memory = 0.0
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
    else:
        # For CPU, we can't easily measure peak memory, so we estimate based on tensor sizes
        estimated_memory = x.numel() * x.element_size() / 1024 / 1024  # MB
        peak_memory = float(
            estimated_memory * 4
        )  # Rough estimate including intermediate tensors

    return avg_time, peak_memory
