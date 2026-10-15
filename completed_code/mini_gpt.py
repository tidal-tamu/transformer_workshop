"""
Mini-GPT Workshop: TUNED REFERENCE VERSION
Same model as starter_code.py, plus the tricks that squeeze out more accuracy without
more compute: dropout, weight decay, learning-rate decay, and early stopping.

Run starter_code.py first, then run this and compare the val loss and the plots.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =========================
# FEATURE FLAGS
# =========================
USE_ATTENTION = True
USE_FFWD      = True
USE_RESIDUAL  = True

# =========================
# HYPERPARAMETERS
# =========================
DATA_FILE     = 'France_Wikipedia_Clean_Text.txt'   # any plain-text file in the workshop folder
PROMPT        = "The capital of France is"   # text the model continues (only use characters from the data!)
block_size    = 32     # context length: how many characters the model can see at once
batch_size    = 32     # how many sequences we train on in parallel
n_embd        = 64     # size of each character's vector
n_head        = 4      # number of attention heads (n_embd must be divisible by n_head)
n_layer       = 3      # number of transformer blocks stacked on top of each other
max_iters     = 2000   # training steps
eval_interval = 250    # print loss every N steps
learning_rate = 5e-3   # starting learning rate (decays toward 0 over training, see below)
dropout       = 0.1    # fraction of activations randomly zeroed during training (fights memorizing)
weight_decay  = 0.1    # gently shrinks weights toward 0 every step (also fights memorizing)
patience      = 3      # stop early if val loss hasn't improved for this many evals
temperature   = 0.8    # sampling "creativity": low = safe & repetitive, high = chaotic
SHOW_PLOTS    = True   # pop up the training plots at the end (always saved to training_plots.png)
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
torch.manual_seed(1337)

# =========================
# DATA: characters -> integers
# =========================
# A language model only sees numbers, so give every unique character an ID.
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, '..', DATA_FILE), 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

missing = set(PROMPT) - set(chars)
assert not missing, f"PROMPT uses characters that never appear in {DATA_FILE}: {missing}"

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
        out[split] = losses.mean().item()
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
        self.dropout = nn.Dropout(dropout)

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
        self.last_attention = wei.detach()   # saved only so we can plot it at the end
        wei = self.dropout(wei)        # randomly block some connections so no single one is relied on
        return wei @ v                 # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    """ Several heads in parallel: one might track vowels, another line breaks, etc. """

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    """ Attention = communication between characters. Feed-forward = each character thinking alone. """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd),
                                 nn.Dropout(dropout))

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


def layer_of(param_name):
    """ Group parameter names into layers for the gradient plot: 'blocks.0.sa...' -> 'block 1'. """
    parts = param_name.split('.')
    return f"block {int(parts[1]) + 1}" if parts[0] == 'blocks' else parts[0]


# =========================
# TRAINING
# =========================
model = MiniGPT().to(device)
print(f"device: {device} | params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
print(f"flags: attention={USE_ATTENTION} ffwd={USE_FFWD} residual={USE_RESIDUAL}\n")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# LEARNING RATE DECAY: take big steps early, then smaller and smaller ones to settle into a
# good minimum instead of bouncing around it (cosine curve from learning_rate down to 0).
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)

# Recorded for the plots at the end
eval_history = []                              # (step, train loss, val loss)
step_losses, lr_history = [], []               # every training step
grad_layers = ['token_embedding'] + [f"block {i + 1}" for i in range(n_layer) if USE_ATTENTION] + ['lm_head']
grad_steps, grad_history = [], {name: [] for name in grad_layers}   # every 10 steps

# EARLY STOPPING: train loss always goes down, but once val loss stops improving the model
# is just memorizing the training text. Remember the best weights and stop when we pass them.
best_val, best_state, bad_evals = float('inf'), None, 0
for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        eval_history.append((it, losses['train'], losses['val']))
        print(f"step {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
        if losses['val'] < best_val:
            best_val, bad_evals = losses['val'], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_evals += 1
            if bad_evals >= patience:
                print(f"val loss hasn't improved in {patience} evals - stopping early")
                break
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()   # BACKPROP: compute how much each weight contributed to the loss

    # How big is the gradient reaching each layer? (see the "Backprop" plot)
    # Measured every 10 steps - measuring every step would double the training time.
    if it % 10 == 0:
        sq = {name: 0.0 for name in grad_layers}
        for name, p in model.named_parameters():
            if p.grad is not None and layer_of(name) in sq:
                sq[layer_of(name)] = sq[layer_of(name)] + p.grad.pow(2).sum()
        grad_steps.append(it)
        for name in grad_layers:
            grad_history[name].append(float(sq[name]) ** 0.5)
    step_losses.append(loss.item())
    lr_history.append(scheduler.get_last_lr()[0])

    optimizer.step()  # GRADIENT DESCENT: nudge every weight a little bit downhill
    scheduler.step()

model.load_state_dict(best_state)   # roll back to the best checkpoint
model.eval()                        # turn dropout OFF for generation
print(f"using best checkpoint (val loss {best_val:.3f})")

# =========================
# GENERATE SOME TEXT
# =========================
print("\n" + "=" * 60)
# The model continues whatever text we give it. Phrase it as a sentence to finish,
# not a question: this is a text continuer, not a chatbot.
context = torch.tensor([encode(PROMPT)], dtype=torch.long, device=device)
out = decode(model.generate(context, max_new_tokens=500)[0].tolist())
# Weak models (especially the bigram at low temperature) love spitting out newline after
# newline, so squash runs of blank lines to keep the output readable.
print(re.sub(r'\n{3,}', '\n\n', out))


# =========================
# PLOTS: watch gradient descent and backprop at work
# =========================
def smooth(steps, values, window=5):
    """ Moving average so the trend is visible through the step-to-step noise. """
    if len(values) < window:
        return steps, values
    return steps[window - 1:], np.convolve(values, np.ones(window) / window, mode='valid')


BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6a64', '#e6e5e0'
BLUE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8})
fig, ((ax_loss, ax_grad), (ax_att, ax_lr)) = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"attention={USE_ATTENTION}  ffwd={USE_FFWD}  residual={USE_RESIDUAL}   |   best val loss {best_val:.3f}",
             color=INK)

# 1. LOSS: gradient descent pushing the error down. Faint line = every noisy mini-batch step.
steps, train_l, val_l = zip(*eval_history)
ax_loss.plot(step_losses, color=BLUE, alpha=0.15, linewidth=1)
ax_loss.plot(steps, train_l, color=BLUE, linewidth=2, marker='o', markersize=5, label='train')
ax_loss.plot(steps, val_l, color=ORANGE, linewidth=2, marker='o', markersize=5, label='val')
ax_loss.set(title='Gradient descent: loss over training', xlabel='step', ylabel='cross-entropy loss')
ax_loss.legend(frameon=False)

# 2. BACKPROP: gradient size reaching each layer. Light = near the input, dark = near the output.
#    Without residual connections, the gradients reaching the first layers collapse early in
#    training (vanishing gradients) and the loss stalls on a plateau. Compare USE_RESIDUAL on/off!
line_ends = []
for i, name in enumerate(grad_layers):
    color = BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(len(grad_layers) - 1, 1))]
    xs, ys = smooth(grad_steps, grad_history[name])
    ax_grad.plot(xs, ys, color=color, linewidth=2, label=name)
    line_ends.append([np.log10(ys[-1]), name, xs[-1]])
ax_grad.set_yscale('log')
ax_grad.margins(x=0.15)   # room for the line labels
lo, hi = np.log10(ax_grad.get_ylim())
line_ends.sort()
for prev, cur in zip(line_ends, line_ends[1:]):   # nudge labels apart so they don't overlap
    cur[0] = max(cur[0], prev[0] + 0.045 * (hi - lo))
for y, name, x in line_ends:
    ax_grad.annotate(name, (x, 10 ** y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK)
ax_grad.set(title='Backprop: gradient size reaching each layer', xlabel='step', ylabel='gradient norm (log)')
ax_grad.legend(frameon=False, fontsize=8, title='input -> output', title_fontsize=8)

# 3. ATTENTION: for each character (row), how much it looks at each earlier character (column).
#    The empty upper triangle is the causal mask. Try other layers/heads by changing the indices.
if USE_ATTENTION:
    sample = out[:block_size]
    with torch.no_grad():
        model(torch.tensor([encode(sample)], dtype=torch.long, device=device))
    att = model.blocks[0].sa.heads[0].last_attention[0].cpu().numpy()
    labels = [{'\n': '\\n', ' ': '·'}.get(c, c) for c in sample]
    cmap = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#5598e7', '#0d366b'])
    ax_att.imshow(att, cmap=cmap, vmin=0, vmax=1)
    ax_att.set_xticks(range(len(labels)), labels, fontsize=7)
    ax_att.set_yticks(range(len(labels)), labels, fontsize=7)
    ax_att.grid(False)
    ax_att.set(title='Attention: layer 1, head 1', xlabel='looking at (key)', ylabel='current character (query)')
else:
    ax_att.axis('off')
    ax_att.text(0.5, 0.5, 'Attention is off\n(turn on USE_ATTENTION)', ha='center', va='center', color=MUTED)

# 4. LEARNING RATE: the size of each gradient descent step.
ax_lr.plot(lr_history, color=BLUE, linewidth=2)
ax_lr.set(title='Learning rate (step size) over training', xlabel='step', ylabel='learning rate')
ax_lr.set_ylim(0, max(lr_history) * 1.15)

fig.tight_layout()
fig.savefig(os.path.join(HERE, 'training_plots.png'), dpi=120)
print(f"\nplots saved to {os.path.join(HERE, 'training_plots.png')}")
if SHOW_PLOTS:
    plt.show()
