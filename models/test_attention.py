import torch
import torch.nn.functional as F                
from transformer import ScaledDotProductAttention
import math

# Tiny test
batch_size = 2
seq_len = 5
d_k = 64

q = torch.randn(batch_size, seq_len, d_k)
k = torch.randn(batch_size, seq_len, d_k)
v = torch.randn(batch_size, seq_len, d_k)

# Causal mask: lower triangle = 1, upper = 0
mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

attn = ScaledDotProductAttention(dropout=0.0)
output = attn(q, k, v, mask)

print("Output shape:", output.shape)               # should be [2, 5, 64]
print("Has NaN?", torch.isnan(output).any().item())
print("Has Inf?", torch.isinf(output).any().item())

# Vis Check
print("\nAttention scores example (first batch item):\n")
scores = torch.matmul(q[0], k[0].T) / math.sqrt(d_k)
masked = scores.masked_fill(mask[0,0] == 0, -1e9)
probs = F.softmax(masked, dim=-1)                  
print(probs.round(decimals=3))