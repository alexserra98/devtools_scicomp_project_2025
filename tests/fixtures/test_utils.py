import torch
import tempfile
import os
from typing import Dict, Any, Optional, Tuple


class MockTokenizer:
    """Mock tokenizer for testing purposes."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """Encode text to token IDs."""
        # Simple deterministic encoding based on text content
        tokens = []
        words = text.split()
        for word in words:
            for char in word:
                tokens.append(ord(char) % self.vocab_size)
        return tokens[:200]  # Limit length


def create_test_model_config(
    vocab_size: int = 100,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    attention_type: str = "standard",
    **kwargs,
) -> Dict[str, Any]:
    """Create a test model configuration."""
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "max_seq_len": 128,
        "attention_type": attention_type,
        "dropout": 0.0,  # Disable dropout for consistent testing
    }
    config.update(kwargs)
    return config


def create_test_data(
    batch_size: int = 2,
    seq_len: int = 32,
    vocab_size: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Create test input data."""
    if device is None:
        device = torch.device("cpu")

    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    check_finite: bool = True,
    check_range: Optional[Tuple[float, float]] = None,
):
    """Assert common tensor properties."""
    if expected_shape is not None:
        assert tensor.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {tensor.shape}"
        )

    if check_finite:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"

    if check_range is not None:
        min_val, max_val = check_range
        assert tensor.min().item() >= min_val, (
            f"Tensor minimum {tensor.min().item()} below {min_val}"
        )
        assert tensor.max().item() <= max_val, (
            f"Tensor maximum {tensor.max().item()} above {max_val}"
        )


def create_temporary_file(suffix: str = ".tmp") -> str:
    """Create a temporary file and return its path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        return f.name


def cleanup_temporary_file(filepath: str):
    """Clean up a temporary file."""
    if os.path.exists(filepath):
        os.unlink(filepath)


class ModelTrainerForTesting:
    """Utility class for training models in tests."""

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

    def quick_train(
        self,
        dataset,
        num_epochs: int = 2,
        batch_size: int = 4,
        learning_rate: float = 1e-3,
    ) -> float:
        """Perform quick training for testing purposes."""
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.model.train()
        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, targets=input_ids)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducible testing."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
