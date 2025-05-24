###########################################
# Author - Shreyas Bhavsar                #
# Date - November 2023                    #
###########################################



import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------
# Core modules
# -------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].to(x.device)
        return self.drop(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k, self.h = d_model // h, h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        bs = query.size(0)
        # project and split heads
        qs, ks, vs = [
            lin(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        # scaled dot-product
        scores = (qs @ ks.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.drop(attn)
        x = (attn @ vs).transpose(1, 2).contiguous().view(bs, -1, self.h * self.d_k)
        return self.linears[-1](x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Sublayer(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        return x + self.drop(sublayer_fn(self.norm(x)))


# -------------------------------------------------------------------
# Encoder / Decoder Layers & Stacks
# -------------------------------------------------------------------

def subsequent_mask(size):
    return torch.triu(torch.ones(1, size, size), diagonal=1).eq(0)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([Sublayer(d_model, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, mask))
        return self.sublayers[1](x, self.ff)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout)
        self.src_attn  = MultiHeadAttention(h, d_model, dropout)
        self.ff        = FeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList([Sublayer(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayers[0](x, lambda y: self.self_attn(y, y, y, tgt_mask))
        x = self.sublayers[1](x, lambda y: self.src_attn(y, memory, memory, src_mask))
        return self.sublayers[2](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_k * layer.self_attn.h)

    def forward(self, x, mask):
        for l in self.layers:
            x = l(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.self_attn.d_k * layer.self_attn.h)

    def forward(self, x, memory, src_mask, tgt_mask):
        for l in self.layers:
            x = l(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# -------------------------------------------------------------------
# Embeddings & Generator
# -------------------------------------------------------------------

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


# -------------------------------------------------------------------
# Assemble Transformer
# -------------------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super().__init__()
        # Core stacks
        enc_layer = EncoderLayer(d_model, h, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, h, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        # Embeddings + position
        pe = PositionalEncoding(d_model, dropout)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab),   pe)
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab),   pe)
        # Final projection
        self.generator = Generator(d_model, tgt_vocab)

        # Xavier init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, tgt_mask)


# -------------------------------------------------------------------
# Loss, Data, Training utilities (unchanged)
# -------------------------------------------------------------------

class LabelSmoothing(nn.Module):
    def __init__(self, size, pad, smoothing=0.1):
        super().__init__()
        self.crit = nn.KLDivLoss(reduction="sum")
        self.size, self.pad, self.smoothing = size, pad, smoothing
        self.confidence = 1 - smoothing

    def forward(self, x, target):
        true_dist = x.clone().fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad] = 0
        mask = (target == self.pad).nonzero().squeeze()
        if mask.numel() > 0:
            true_dist.index_fill_(0, mask, 0.0)
        return self.crit(x, true_dist)


class SimpleLossCompute:
    def __init__(self, gen, crit, opt=None):
        self.gen, self.crit, self.opt = gen, crit, opt

    def __call__(self, x, y, norm):
        # x: (batch, seq_len, d_model)
        # y: (batch, seq_len)
        logits = self.gen(x)  # → (batch, seq_len, vocab)

        # Use reshape instead of view:
        logits_flat = logits.reshape(-1, logits.size(-1))  # (batch*seq_len, vocab)
        y_flat      = y.reshape(-1)                        # (batch*seq_len,)

        loss = self.crit(logits_flat, y_flat) / norm

        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

        return loss * norm



def data_gen(V, batch, n_batches, seq_len=10):
    for _ in range(n_batches):
        data = torch.randint(1, V, (batch, seq_len))
        data[:, 0] = 1
        yield Batch(data, data, pad=0)


class Batch:
    def __init__(self, src, tgt, pad=0):
        self.src, self.src_mask = src, (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            mask = (self.tgt != pad).unsqueeze(-2)
            self.tgt_mask = mask & subsequent_mask(self.tgt.size(-1)).to(mask)
            self.ntokens = (self.tgt_y != pad).sum().item()


# -------------------------------------------------------------------
# Example Training + Inference (unchanged)
# -------------------------------------------------------------------

def run_copy_task():
    V = 11
    model = Transformer(V, V, N=2)
    crit = LabelSmoothing(size=V, pad=0, smoothing=0.1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.98), eps=1e-9)
    d_model = model.src_embed[0].d_model
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda step: d_model**-0.5 * (step+1)**-0.5)
    loss_compute = SimpleLossCompute(model.generator, crit, opt)

    for epoch in range(5):
        model.train()
        total = 0
        for i, batch in enumerate(data_gen(V, 32, 30)):
            out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
            loss = loss_compute(out, batch.tgt_y, batch.ntokens)
            total += loss.item()
            sched.step()
        print(f"Epoch {epoch+1}: Avg Loss = {total/(i+1):.4f}")


def example_usage_real_data(src, tgt, src_vocab, tgt_vocab):
    model = Transformer(src_vocab, tgt_vocab)
    model.eval()
    src_mask = (src != 0).unsqueeze(-2)
    tgt_in  = tgt[:, :-1]
    tgt_mask = (tgt_in != 0).unsqueeze(-2) & subsequent_mask(tgt_in.size(-1)).to(src_mask)
    with torch.no_grad():
        logits = model(src, tgt_in, src_mask, tgt_mask)
        return torch.exp(model.generator(logits))

if __name__ == "__main__":
    run_copy_task()
