#model.py

"""
Custom GPT-2 Model Implementation with the following modifications:
- Config loading from dict/JSON
- attention_mask support
- save/load methods for checkpointing
- transformer property for validation compatibility
"""

import json
import math
import inspect
import torch
from torch import nn
from torch.nn import functional as F


class MHA(nn.Module):
    """Multi-Head Attention with causal masking."""
    
    def __init__(self, d_model, num_heads, attn_pdrop, resid_pdrop):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        
        # Build attention mask for padding
        attn_mask = None
        if attention_mask is not None:
            # attention_mask: [B, T] -> [B, 1, 1, T]
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = (1.0 - attn_mask.float()) * -1e9
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_pdrop if self.training else 0,
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    """Feed-Forward Network with GELU activation."""
    
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block with Pre-LayerNorm."""
    
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop, resid_pdrop, ff_dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, num_heads, attn_pdrop, resid_pdrop)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, ff_dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2(nn.Module):
    """
    Custom GPT-2 Model with Pre-LayerNorm.
    
    Args:
        config: dict with model configuration
    """
    
    architecture_name = "Thoth_v1"
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Store config
        self.config = config
        model_cfg = config.get("model", config)
        
        # Extract parameters
        self.vocab_size = model_cfg["vocab_size"]
        self.d_model = model_cfg["d_model"]
        self.n_layer = model_cfg["n_layer"]
        self.n_head = model_cfg["n_head"]
        self.d_ff = model_cfg["d_ff"]
        self.max_seq_len = model_cfg["max_seq_len"]
        
        # Dropout
        drop = model_cfg.get("dropout", {})

        embed_dropout = drop.get("embed", 0.1)
        attn_dropout  = drop.get("attn", 0.1)
        resid_dropout = drop.get("resid", 0.1)
        ff_dropout    = drop.get("ff", 0.1)

        # Embeddings
        self.token_embed = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embed = nn.Embedding(self.max_seq_len, self.d_model)
        self.dropout_embed = nn.Dropout(embed_dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(self.d_model, self.n_head, self.d_ff, 
                  attn_dropout, resid_dropout, ff_dropout)
            for _ in range(self.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(self.d_model)
        
        # LM head (weight tying with token embeddings)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_config(cls, config_path: str):
        """Load model from JSON config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.
        
        Args:
            input_ids: [B, T] token indices
            attention_mask: [B, T] attention mask (1 = attend, 0 = ignore)
            labels: [B, T] target token indices for loss computation
            
        Returns:
            logits: [B, T, V] or [B, V] if no labels
            loss: scalar loss if labels provided, else None
        """
        device = input_ids.device
        B, T = input_ids.size()
        
        # attention_mask: [B, T], 1 for real tokens, 0 for padding
        pos = attention_mask.long().cumsum(dim=1) - 1
        pos = pos.clamp(min=0)

        
        # Embeddings
        tok_emb = self.token_embed(input_ids)
        pos_emb = self.pos_embed(pos)
        x = self.dropout_embed(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Always compute full logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            loss = None
        
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay.
        
        Applies weight decay to 2D+ parameters (weights), not to biases/LayerNorm.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Decay params: {len(decay_params)} tensors, {num_decay:,} parameters")
        print(f"No-decay params: {len(nodecay_params)} tensors, {num_nodecay:,} parameters")
        
        # Use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            betas=betas, 
            **extra_args
        )
        print(f"Using fused AdamW: {use_fused}")
        
        return optimizer

    def save(self, path: str):
        """Save model weights and config."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)

    @classmethod
    def load(cls, path: str, config: dict = None):
        """
        Load model from checkpoint.
        
        Args:
            path: path to checkpoint file
            config: optional config dict (if not saved in checkpoint)
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        if config is None:
            config = checkpoint.get('config')
            if config is None:
                raise ValueError("No config provided and none found in checkpoint")
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    # === Compatibility layer for validation script ===
    
    @property
    def transformer(self):
        """Compatibility property for validation script."""
        return self._TransformerWrapper(self)
    
    class _TransformerWrapper:
        """Wrapper to provide HuggingFace-like transformer interface."""
        
        def __init__(self, model):
            self.model = model
        
        def __call__(self, input_ids, attention_mask=None):
            """
            Forward pass returning hidden states.
            
            Returns:
                Object with last_hidden_state attribute
            """
            device = input_ids.device
            B, T = input_ids.size()
            
            tok_emb = self.model.token_embed(input_ids)

            if attention_mask is None:
                # fallback to old behavior
                pos = torch.arange(0, T, dtype=torch.long, device=device)
                pos_emb = self.model.pos_embed(pos)
            else:
                # per-sample positions
                pos = attention_mask.long().cumsum(dim=1) - 1
                pos = pos.clamp(min=0)
                pos_emb = self.model.pos_embed(pos)

            x = self.model.dropout_embed(tok_emb + pos_emb)

            for block in self.model.blocks:
                x = block(x, attention_mask)
            
            x = self.model.ln_f(x)
            
            return self._Output(x)
        
        class _Output:
            def __init__(self, hidden_state):
                self.last_hidden_state = hidden_state

    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

