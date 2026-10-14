"""
Mini-GPT Workshop: From Bigram to Transformer (COMPLETED VERSION)
Train a tiny character-level GPT on Shakespeare, one idea at a time.

Workflow:
    1. Run with all USE_* flags False  -> a plain bigram model (gibberish)
    2. Turn on USE_ATTENTION           -> characters can look at the past
    3. Turn on USE_FFWD                -> each character "thinks" about what it saw
    4. Turn on USE_RESIDUAL            -> deep stacks actually train (real words!)
Expected val loss (2000 steps): ~2.50 -> ~2.16 -> ~2.38 (worse! deep nets without
residuals are hard to train) -> ~1.82 (character names and real words appear).
"""

import os
import re
import torch
import torch.nn as nn
from torch.nn import functional as F

# =========================
# FEATURE FLAGS (flip these one at a time!)
# =========================
USE_ATTENTION = True
USE_FFWD      = True
USE_RESIDUAL  = True

# =========================
# HYPERPARAMETERS
# =========================
block_size    = 32     # context length: how many characters the model can see at once
batch_size    = 32     # how many sequences we train on in parallel
n_embd        = 64     # size of each character's vector
n_head        = 4      # number of attention heads (each one is n_embd // n_head wide)
n_layer       = 3      # number of transformer blocks stacked on top of each other
max_iters     = 2000   # training steps
eval_interval = 250    # print loss every N steps
learning_rate = 3e-3
temperature   = 0.8    # sampling "creativity": low = safe & repetitive, high = chaotic
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

torch.manual_seed(1337)

# =========================
# DATA: characters -> integers
# =========================
# A language model only sees numbers, so give every unique character an ID.
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shakespeare_text.txt')
with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]   # hold out 10% to check we aren't just memorizing


def get_batch(split):
    """
    Grab random chunks of text. The target for each position is simply the NEXT character:
        x = "To be or"
        y = "o be or "
    So y is x shifted one character to the right.
    """
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# =========================
# SELF-ATTENTION: how characters talk to each other
# =========================
class Head(nn.Module):
    """
    Every character emits three vectors:
        query (Q): "what am I looking for?"
        key   (K): "what do I contain?"
        value (V): "what will I share if you pick me?"
    Attention score between two characters = Q . K  (high when they're a good match).
    Each character's output = a weighted average of the V's it attends to.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular matrix of 1s: row t has 1s only for positions 0..t
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)   # each (B, T, head_size)

        # Scaled dot-product: divide by sqrt(head_size) so scores don't explode and
        # make softmax too "peaky" (one character hogging all the attention).
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)

        # CAUSAL MASK - this is what makes it a DECODER (GPT-style).
        # A character may not look at the future, otherwise it could just read the answer.
        # Setting future scores to -inf makes their softmax weight exactly 0.
        # (An ENCODER, like BERT, has no mask: every token sees the whole sentence.)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        wei = F.softmax(wei, dim=-1)   # each row sums to 1 -> attention weights
        return wei @ v                 # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """ Several heads in parallel: one might track vowels, another line breaks, etc. """

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    """ Attention = communication between characters. Feed-forward = each character thinking alone. """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ One transformer block: communicate (attention), then compute (feed-forward). """

    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = FeedForward() if USE_FFWD else nn.Identity()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        if USE_RESIDUAL:
            # RESIDUAL CONNECTION: add the block's output ON TOP of its input instead of
            # replacing it. Gradients get a "highway" straight back through the network,
            # which is what lets deep stacks train. (LayerNorm keeps values well-scaled.)
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
        else:
            x = self.sa(x)
            x = self.ffwd(x)
        return x


# =========================
# THE MODEL
# =========================
class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)        # WHAT the character is
        self.position_embedding = nn.Embedding(block_size, n_embd)     # WHERE it is in the context
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)]) if USE_ATTENTION else nn.Identity()
        self.ln_f = nn.LayerNorm(n_embd) if USE_RESIDUAL else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)                   # vector -> score for every next char

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        if USE_ATTENTION:
            # Attention has no built-in sense of order, so we tell it positions explicitly.
            # Without attention this is a pure BIGRAM: the next char depends only on the current one.
            x = x + self.position_embedding(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # The model only knows block_size positions, so keep the LAST block_size characters.
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            # Only the prediction at the final position matters. Dividing by temperature
            # sharpens (<1) or flattens (>1) the distribution before sampling.
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =========================
# TRAINING
# =========================
model = MiniGPT().to(device)
print(f"device: {device} | params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
print(f"flags: attention={USE_ATTENTION} ffwd={USE_FFWD} residual={USE_RESIDUAL}\n")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# =========================
# GENERATE SOME SHAKESPEARE
# =========================
print("\n" + "=" * 60)
# The model continues whatever text we give it (a "prompt"). Try your own!
prompt = "ROMEO:\n"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
out = decode(model.generate(context, max_new_tokens=500)[0].tolist())
# Weak models (especially the bigram at low temperature) love spitting out newline after
# newline, so squash runs of blank lines to keep the output readable.
print(re.sub(r'\n{3,}', '\n\n', out))
