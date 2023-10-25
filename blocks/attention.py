import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module) :
    def __init__(self, d_model, h, dropout) : 
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        assert d_model % h == 0
        self.dropout = nn.Dropout(dropout)
        # Define the Q, K, and V matrices
        self.qvk = nn.Linear(d_model, d_model*3)
        self.output_proj = nn.Linear(d_model, d_model)
    @staticmethod
    def computeAttention(query, key, value, dropout=None, mask=None) : 
        d_k = query.shape[-1]
        # compute Q*K.T/sqrt(d_k)
        attentions_scores = (query@key.transpose(-1,-2))/math.sqrt(d_k)
        if mask is not None : 
            attentions_scores = attentions_scores.masked_fill_(mask, 1e9)
        # compute softmax with respect to the model dimension
        attentions_scores = F.softmax(attentions_scores, dim=-1)
        if dropout is not None : 
            attentions_scores = dropout(attentions_scores)
        return attentions_scores @ value, attentions_scores
        
    # q=k=v=x in self attention, this is why we only pass one parameter x
    def forward(self, x, causal_mask) : 
        batch_size, seq_len, d_model = x.shape
        # query, key, value are of shape (batch_size, seq_len, d_model)
        query, key, value = self.qvk(x).chunk(3, dim=-1)
        # set query, key, value shapes to : (batch_size, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).permute(0,2,1,3)
        key = key.view(batch_size, seq_len, self.h, self.d_k).permute(0,2,1,3)
        value = value.view(batch_size, seq_len, self.h, self.d_k).permute(0,2,1,3)
        # we compute attention, x of shape (batch_size, h, seq_len, d_k)
        x, attention_scores = SelfAttention.computeAttention(query, key, value, self.dropout, causal_mask)
        # we reshape x to (batch_size, seq_len, h, d_k)
        x = x.permute(0,2,1,3)
        # we merge back h and d_k to d_model
        print(x.shape)
        x = x.contiguous().view(batch_size,seq_len,d_model)
        # we project x with the output projection, size of x (batch_size, seq_len, d_model)
        x = self.output_proj(x)

        return x
        
att = SelfAttention(200,8,0.1)
att(torch.rand(2,50,200), torch.zeros((50,50), dtype=torch.bool)).shape

class CrossAttention(nn.Module) :
    def __init__(self, d_model, h, dropout) : 
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        assert d_model % h == 0
        self.dropout = nn.Dropout(dropout)
        # Define the Q, K, and V matrices
        self.kv = nn.Linear(d_model, d_model*2)
        self.q = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
    @staticmethod
    def computeAttention(query, key, value, dropout=None, mask=None) : 
        d_k = query.shape[-1]
        # compute Q*K.T/sqrt(d_k)
        attentions_scores = (query@key.transpose(-1,-2))/math.sqrt(d_k)
        if mask is not None : 
            attentions_scores = attentions_scores.masked_fill_(mask, 1e9)
        # compute softmax with respect to the model dimension
        attentions_scores = F.softmax(attentions_scores, dim=-1)
        if dropout is not None : 
            attentions_scores = dropout(attentions_scores)
        return attentions_scores @ value, attentions_scores
        
    # in cross attention q is different, and k=v, we only pass two parameters : query and x
    def forward(self, query, x, causal_mask) : 
        batch_size, seq_len, d_model = x.shape
        # query, key, value are of shape (batch_size, seq_len, d_model)
        key, value = self.kv(x).chunk(2, dim=-1)
        query = self.q(x)
        # set query, key, value shapes to : (batch_size, h, seq_len, d_k)
        query = query.view(batch_size, seq_len, self.h, self.d_k).permute(0,2,1,3)
        key = key.view(batch_size, seq_len, self.h, self.d_k).permute(0,2,1,3)
        value = value.view(batch_size, seq_len, self.h, self.d_k).permute(0,2,1,3)
        # we compute attention, x of shape (batch_size, h, seq_len, d_k)
        x, attention_scores = SelfAttention.computeAttention(query, key, value, self.dropout, causal_mask)
        # we reshape x to (batch_size, seq_len, h, d_k)
        x = x.permute(0,2,1,3)
        # we merge back h and d_k to d_model
        x = x.contiguous().view(batch_size,seq_len,d_model)
        # we project x with the output projection, size of x (batch_size, seq_len, d_model)
        x = self.output_proj(x)
        return x
    
att = CrossAttention(200,8,0.1)
att(torch.rand(2,50,200), torch.rand(2,50,200),torch.zeros((50,50), dtype=torch.bool)).shape