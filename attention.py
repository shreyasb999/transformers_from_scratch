import copy
import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################################
# Utility helpers
###############################################################

def clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size: int) -> torch.Tensor:
    """Mask out subsequent positions (for decoder self-attention)."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

###############################################################
# Core components
###############################################################

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by number of heads"
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn: Optional[torch.Tensor] = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # Linear projections: d_model -> h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # Scaled dot-product attention
        x, self.attn = self.scaled_dot_product_attention(query, key, value, mask)
        # Concat heads
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class SublayerConnection(nn.Module):
    """Residual connection followed by layer normalization"""

    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """One encoder layer."""

    def __init__(self, d_model: int, self_attn: MultiHeadAttention, feed_forward: PositionwiseFeedForward, dropout: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    """One decoder layer."""

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention,
        feed_forward: PositionwiseFeedForward,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer[1](x, lambda _x: self.src_attn(_x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """Transformer encoder composed of N layers."""

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Transformer decoder composed of N layers."""

    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Embeddings(nn.Module):
    """Token embedding + scaling."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    """Final linear + log-softmax."""

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """Full Transformer model."""

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: nn.Module,
        tgt_embed: nn.Module,
        generator: Generator,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)


#################################################################
# Model construction helper
#################################################################

def make_model(
    src_vocab: int,
    tgt_vocab: int,
    N: int = 6,
    d_model: int = 512,
    d_ff: int = 2048,
    h: int = 8,
    dropout: float = 0.1,
) -> Transformer:
    """Constructs a Transformer from hyperparameters."""
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        Decoder(DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), copy.deepcopy(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), copy.deepcopy(position)),
        Generator(d_model, tgt_vocab),
    )
    # Parameter init: Glorot / Xavier uniform
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

#################################################################
# Label smoothing and loss computation
#################################################################

class LabelSmoothing(nn.Module):
    """Label smoothing using KL divergence."""

    def __init__(self, size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist)


class SimpleLossCompute:
    """Compute loss, backpropagate, and step optimizer."""

    def __init__(self, generator: Generator, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm: float,
    ) -> torch.Tensor:
        x = self.generator(x)
        loss = self.criterion(
            x.contiguous().view(-1, x.size(-1)),
            y.contiguous().view(-1),
        ) / norm
        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        return loss * norm

#################################################################
# Example data generation and batch class for copy task
#################################################################

def data_gen(V: int, batch_size: int, nbatches: int, seq_len: int = 10):
    """Generate random data for a src->tgt copy task."""
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, seq_len))
        data[:, 0] = 1  # start symbol
        yield Batch(data, data, pad=0)


class Batch:
    """Holds a batch of data with masks."""

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor, pad: int = 0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int) -> torch.Tensor:
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask

#################################################################
# Example training and usage
#################################################################
def run_copy_task():
    """Train Transformer on toy copy task."""
    V = 11
    model = make_model(V, V, N=2)
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9
    )
    # Use Noam schedule: d_model retrieved from model
    d_model = model.src_embed[0].d_model
    def lr_lambda(step):
        return d_model ** -0.5 * (step + 1) ** -0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_compute = SimpleLossCompute(model.generator, criterion, optimizer)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for i, batch in enumerate(data_gen(V, batch_size=32, nbatches=30)):
            out = model(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            loss = loss_compute(out, batch.tgt_y, batch.ntokens)
            total_loss += loss.item()
            scheduler.step()
        print(f"Epoch {epoch+1}: Avg Loss = {total_loss/(i+1):.4f}")


def example_usage_real_data(src_tensor, tgt_tensor, src_vocab_size, tgt_vocab_size):
    """Example of using the Transformer on real data tensors.
    src_tensor, tgt_tensor should be torch.LongTensors of shape [batch, seq_len].
    """
    # Build model
    model = make_model(src_vocab_size, tgt_vocab_size)
    model.eval()
    # Create masks
    src_mask = (src_tensor != 0).unsqueeze(-2)
    tgt_input = tgt_tensor[:, :-1]
    tgt_mask = (tgt_input != 0).unsqueeze(-2) & subsequent_mask(tgt_input.size(-1)).type_as(src_mask)
    # Forward pass
    with torch.no_grad():
        logits = model(src_tensor, tgt_input, src_mask, tgt_mask)
        # Convert to probabilities
        probs = torch.exp(model.generator(logits))
    return probs


if __name__ == "__main__":
    print("Running copy task example...")
    run_copy_task()
    # For real data, call example_usage_real_data with your tensors
    # e.g., probs = example_usage_real_data(src_batch, tgt_batch, vocab1, vocab2)
