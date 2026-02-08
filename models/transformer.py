import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):
    """
    Core attention mechanism.
    Implements QK^T / sqrt(d_k) + optional causal masking + softmax + V
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout =nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None           
    ) -> torch.Tensor: 
        d_k = q.size(-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
    
        output = torch.matmul(attn_probs, v)
    
        return output

class MultiHeadAttention(nn.Module):
    """
    Multihead attention wrapper.
    Splits d_model into num_heads * d_k, computers attention in parallel, then concatenates
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        #Linear projections
        self.q_proj = nn.Linear(d_model, d_model)   
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,      
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None 
    ) -> torch.Tensor:
        
        batch_size, seq_len, _ = query.shape
        # Reshape tensor
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Parallel masking
        if mask is not None:
            mask = mask.unsqueeze(1)

        attn_output = self.attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.out_proj(attn_output)
        return self.dropout(output)
    
if __name__ == "__main__":
    torch.manual_seed(42)  
    
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.0)
    
    output = mha(x, x, x, mask)
    
    print("MultiHeadAttention output shape:", output.shape)  # should be [2, 8, 64]
    print("Has NaN?", torch.isnan(output).any().item())
    print("Has Inf?", torch.isinf(output).any().item())
    print("Output mean / std:", output.mean().item(), output.std().item())
