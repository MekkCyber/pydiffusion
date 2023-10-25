import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class ClipEmbedding(nn.Module) : 
    def __init__(self, vocab_size, emb_dim, n_token) : 
        super().__init__() 
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(n_token, emb_dim))

    def forward(self, x) : 
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, emb_dim)
        return self.embed(x)+self.positional_encoding
    
class ClipLayer(nn.Module) : 
    def __init__(self, emb, h, dropout=0.1) : 
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb)
        self.layernorm2 = nn.LayerNorm(emb)
        self.linear1 = nn.Linear(emb, emb*4)
        self.linear2 = nn.Linear(emb*4, emb)
        self.attention = SelfAttention(emb, h, dropout)

    def forward(self, x) : 
        residual = x
        x = self.layernorm1(x)
        x = self.attention(x)
        x += residual
        residual = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = F.sigmoid(1.702*x)
        x = self.linear2(x)
        x += residual

        return x
    

class Clip(nn.Module) : 
    def __init__(self, vocab_size, emb_dim=768, n_token=77) : 
        super().__init__() 
        self.embedding = ClipEmbedding(vocab_size, emb_dim, n_token)
        self.layers = [ClipLayer(emb_dim, 12) for _ in range(12)]
        self.layernorm = nn.LayerNorm(emb_dim)
    def forward(self, tokens) :
        tokens = tokens.type(torch.long)
        state = self.embedding(tokens)
        for layer in self.layers: 
            state = layer(state)
        output = self.layernorm(state)
        return output
    
model = Clip(10000, n_token=4)
model(torch.rand(1,4)).shape