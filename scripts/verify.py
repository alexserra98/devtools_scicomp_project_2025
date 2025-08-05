import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality of attention mechanisms."""
    print("Testing basic functionality...")
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from src.attention_implementations import StandardAttention, SparseAttention, FlashAttention
        print("✓ Successfully imported attention implementations")
    except ImportError as e:
        print(f"✗ Failed to import attention implementations: {e}")
        return False
    
    try:
        from src.transformer_model import TransformerLM
        print("✓ Successfully imported transformer model")
    except ImportError as e:
        print(f"✗ Failed to import transformer model: {e}")
        return False
    
    # Test basic attention functionality
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size, seq_len, d_model, num_heads = 2, 16, 64, 4
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # Test standard attention
    try:
        attention = StandardAttention(d_model, num_heads).to(device)
        output = attention(x)
        assert output.shape == x.shape
        print("✓ Standard attention working")
    except Exception as e:
        print(f"✗ Standard attention failed: {e}")
        return False
    
    # Test sparse attention
    try:
        sparse_attention = SparseAttention(d_model, num_heads, sparsity_pattern='local', window_size=8).to(device)
        output = sparse_attention(x)
        assert output.shape == x.shape
        print("✓ Sparse attention working")
    except Exception as e:
        print(f"✗ Sparse attention failed: {e}")
        return False
    
    # Test flash attention
    try:
        flash_attention = FlashAttention(d_model, num_heads, block_size=8).to(device)
        output = flash_attention(x)
        assert output.shape == x.shape
        print("✓ Flash attention working")
    except Exception as e:
        print(f"✗ Flash attention failed: {e}")
        return False
    
    # Test transformer model
    try:
        model = TransformerLM(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            attention_type='standard'
        ).to(device)
        
        input_ids = torch.randint(0, 100, (1, 10), device=device)
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert outputs['logits'].shape == (1, 10, 100)
        print("✓ Transformer model working")
    except Exception as e:
        print(f"✗ Transformer model failed: {e}")
        return False
    
    print("✓ All basic tests passed!")
    return True

def main():
    print("=" * 50)
    print("QUICK VERIFICATION TEST")
    print("=" * 50)
    
    success = test_basic_functionality()
    
    if success:
        print("\n✓ All tests passed! The implementation is working correctly.")
        print("\nYou can now run:")
        print("  python scripts/train.py --quick-test")
        print("  python scripts/profiler.py")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
