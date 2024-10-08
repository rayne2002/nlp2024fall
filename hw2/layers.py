import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import clones
from torch.nn.functional import log_softmax


class LayerNorm(nn.Module):
    "Construct a layernorm module - https://arxiv.org/abs/1607.06450"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
    
class SublayerConnection(nn.Module):
    """
    A residual connection (https://arxiv.org/abs/1512.03385) followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        print("Shape of input x:", x.shape)  # Print the initial shape of x

        norm_x = self.norm(x)
        print("Shape after LayerNorm:", norm_x.shape)  # Check the shape after normalization

        sublayer_x = sublayer(norm_x)
        print("Shape after sublayer:", sublayer_x.shape)  # Check the shape after the sublayer

        dropout_x = self.dropout(sublayer_x)
        print("Shape after Dropout:", dropout_x.shape)  # Check the shape after applying dropout

        output = x + dropout_x
        print("Shape of output:", output.shape)  # Final output shape after addition

        return output
        # return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attention and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
    
def attention(query, key, value, mask=None, dropout=None):
    # Your code here

   # Step 1: Calculate the dot product between queries and keys (transpose of keys)
    d_k = query.size(-1)  # Get the dimension of the keys
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled by sqrt(d_k)
    
    # Step 2: Apply the mask (if any) by filling the masked positions with a large negative number
    if mask is not None:
    # Ensure that the mask has the same shape as the scores
        mask = mask.unsqueeze(1)  # Add a new dimension at index 1 for the heads
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Step 3: Apply the softmax function to get the attention weights
    attn = F.softmax(scores, dim=-1)
    
    # Step 4: Apply dropout (if any)
    if dropout is not None:
        attn = dropout(attn)
    
    # Step 5: Multiply the attention weights by the value matrix to get the final output
    output = torch.matmul(attn, value)
    
    return output, attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        # Your code here
        "Initialize the multi-headed attention mechanism."
        super(MultiHeadedAttention, self).__init__()
        self.h = h  # Number of attention heads
        self.d_model = d_model  # Dimension of the model (input/output)
        self.d_k = d_model // h  # Dimension per head
        self.d_v = d_model // h  # Dimension per head
        
        # Ensure that d_model is divisible by the number of heads
        assert d_model % h == 0

        # Linear layers for projecting the input query, key, and value matrices
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])

        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # Your code here
        "Implements the forward pass of multi-headed attention."
        if mask is not None:
            # Same mask applied to all heads.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # Step 1: Linear projections for query, key, value
        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        
        # Step 2: Apply the attention function (attention() defined earlier)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Step 3: Concatenate the results from all heads and apply a final linear projection
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        
        # Step 4: Apply final linear layer to combine the outputs
        return self.linears[-1](x)
    
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
    
class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

    

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())    

    