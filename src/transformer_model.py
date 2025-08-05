import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from typing import Optional, Dict, Any


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create a causal (lower triangular) mask for autoregressive models."""
    return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:  # Handle odd d_model
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].size(1)])
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe, persistent=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer block with configurable attention mechanism."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_type: str = 'standard',
        dropout: float = 0.1,
        **attention_kwargs
    ):
        super().__init__()
        
        # Import attention implementations here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from attention_implementations import StandardAttention, SparseAttention, FlashAttention
        
        # Create attention layer based on type
        if attention_type == 'standard':
            self.attention = StandardAttention(d_model, num_heads, dropout)
        elif attention_type == 'sparse':
            self.attention = SparseAttention(d_model, num_heads, dropout=dropout, **attention_kwargs)
        elif attention_type == 'flash':
            self.attention = FlashAttention(d_model, num_heads, dropout=dropout, **attention_kwargs)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


class TransformerLM(pl.LightningModule):
    """
    Transformer Language Model with configurable attention mechanisms.
    
    This model can be used to compare different attention implementations
    on language modeling tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 2048,
        attention_type: str = 'standard',
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        warmup_steps: int = 4000,
        weight_decay: float = 0.01,
        **attention_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.attention_type = attention_type
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attention_type=attention_type,
                dropout=dropout,
                **attention_kwargs
            )
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Share weights between embedding and output layer
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.size()
        
        # Create causal mask
        mask = create_causal_mask(seq_len, input_ids.device)
        
        # Token embeddings + positional encoding
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer norm and projection
        x = self.norm(x)
        logits = self.lm_head(x)
        
        outputs = {'logits': logits}
        
        if targets is not None:
            # Compute loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            outputs['loss'] = loss
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        outputs = self(input_ids, targets=input_ids)
        loss = outputs['loss']
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        outputs = self(input_ids, targets=input_ids)
        loss = outputs['loss']
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Create learning rate scheduler
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / self.hparams.warmup_steps
            else:
                return (self.hparams.warmup_steps / step) ** 0.5
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop context if needed
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            outputs = self(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get the model size in megabytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / 1024 / 1024
