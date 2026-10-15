"""
Mini-GPT Workshop: From Bigram to Transformer

Fill in each ___ blank, then flip the flags one at a time and rerun:
    1. All flags False    -> bigram         (blanks: get_batch, generate)
    2. USE_ATTENTION      -> attention      (blanks: Head)
    3. USE_FFWD           -> feed-forward   (better or worse? why?)
    4. USE_RESIDUAL       -> residuals      (blanks: Block)
Then tune the hyperparameters. Answers: completed_code/mini_gpt.py
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
USE_ATTENTION = False
USE_FFWD      = False
USE_RESIDUAL  = False

# =========================
# HYPERPARAMETERS
# =========================
DATA_FILE     = 'France_Wikipedia_Clean_Text.txt'
PROMPT        = "The capital of France is"
block_size    = 32     # context length (characters)
batch_size    = 32     # sequences per step
n_embd        = 64     # vector size per character
n_head        = 4      # attention heads (must divide n_embd)
n_layer       = 3      # transformer blocks
max_iters     = 2000   # training steps
eval_interval = 250
learning_rate = 3e-3
dropout       = 0.1    # fraction of activations zeroed during training (fights memorizing)
temperature   = 0.8    # low = safe, high = chaotic
SHOW_PLOTS    = True
PLOT_HEADS    = [(1, 1), (n_layer, 1)]   # (layer, head) pairs to plot, counting from 1
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
torch.manual_seed(1337)

# =========================
# DATA: characters -> integers
# =========================
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, DATA_FILE), 'r', encoding='utf-8') as f:
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
train_data, val_data = data[:n], data[n:]   # 10% held out for validation


def get_batch(split):
    """ x = "To be or", y = "o be or " -> the target is the NEXT character. """
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
# SELF-ATTENTION
# =========================
class Head(nn.Module):
    """
    Query = "what am I looking for?", Key = "what do I contain?", Value = "what do I share?"
    Each character's output is a weighted average of the Values it attends to.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)

        # softmax smoothing
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # divide by sqrt(head_size)

        # no peeking into the future (prevents cheating) - this is also called a causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # turn each row into weights that sum to 1 (how important each character is)
        wei = F.softmax(wei, dim=-1)
        self.last_attention = wei.detach()   # for the plot
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """ Several heads in parallel, each free to focus on something different. """

    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    """ Attention = characters talking to each other. Feed-forward = each one thinking alone. """

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
            # add the output on top of the input
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
        self.token_embedding = nn.Embedding(vocab_size, n_embd)      # WHAT the character is
        self.position_embedding = nn.Embedding(block_size, n_embd)   # WHERE it is
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)]) if USE_ATTENTION else nn.Identity()
        self.ln_f = nn.LayerNorm(n_embd) if USE_RESIDUAL else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)                 # score for every next character

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)
        if USE_ATTENTION:   # without attention, this is a pure bigram model
            x = x + self.position_embedding(torch.arange(T, device=device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def layer_of(param_name):
    parts = param_name.split('.')
    return f"block {int(parts[1]) + 1}" if parts[0] == 'blocks' else parts[0]


# =========================
# TRAINING
# =========================
model = MiniGPT().to(device)
print(f"device: {device} | params: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")
print(f"flags: attention={USE_ATTENTION} ffwd={USE_FFWD} residual={USE_RESIDUAL}\n")
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

eval_history = []
step_losses = []
grad_layers = ['token_embedding'] + [f"block {i + 1}" for i in range(n_layer) if USE_ATTENTION] + ['lm_head']
grad_steps, grad_history = [], {name: [] for name in grad_layers}

for it in range(max_iters + 1):
    if it % eval_interval == 0:
        losses = estimate_loss()
        eval_history.append((it, losses['train'], losses['val']))
        print(f"step {it:5d} | train loss {losses['train']:.3f} | val loss {losses['val']:.3f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()   # BACKPROP: how much did each weight contribute to the loss?

    if it % 10 == 0:   # record gradient size per layer for the plot
        sq = {name: 0.0 for name in grad_layers}
        for name, p in model.named_parameters():
            if p.grad is not None and layer_of(name) in sq:
                sq[layer_of(name)] = sq[layer_of(name)] + p.grad.pow(2).sum()
        grad_steps.append(it)
        for name in grad_layers:
            grad_history[name].append(float(sq[name]) ** 0.5)
    step_losses.append(loss.item())

    optimizer.step()  # GRADIENT DESCENT: nudge every weight downhill

model.eval()
final_val = eval_history[-1][2]

# =========================
# GENERATE SOME TEXT
# =========================
print("\n" + "=" * 60)
context = torch.tensor([encode(PROMPT)], dtype=torch.long, device=device)
out = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(re.sub(r'\n{3,}', '\n\n', out))   # squash runs of blank lines


# =========================
# PLOTS
# =========================
def smooth(steps, values, window=5):
    if len(values) < window:
        return steps, values
    return steps[window - 1:], np.convolve(values, np.ones(window) / window, mode='valid')


BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1f1f1e', '#6b6a64', '#e6e5e0'
BLUE_RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
plt.rcParams.update({'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8})
fig, ((ax_loss, ax_grad), (ax_att1, ax_att2)) = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(f"attention={USE_ATTENTION}  ffwd={USE_FFWD}  residual={USE_RESIDUAL}   |   final val loss {final_val:.3f}",
             color=INK)

steps, train_l, val_l = zip(*eval_history)
ax_loss.plot(step_losses, color=BLUE, alpha=0.15, linewidth=1)
ax_loss.plot(steps, train_l, color=BLUE, linewidth=2, marker='o', markersize=5, label='train')
ax_loss.plot(steps, val_l, color=ORANGE, linewidth=2, marker='o', markersize=5, label='val')
ax_loss.set(title='Gradient descent: loss over training', xlabel='step', ylabel='cross-entropy loss')
ax_loss.legend(frameon=False)

line_ends = []
for i, name in enumerate(grad_layers):
    color = BLUE_RAMP[round(i * (len(BLUE_RAMP) - 1) / max(len(grad_layers) - 1, 1))]
    xs, ys = smooth(grad_steps, grad_history[name])
    ax_grad.plot(xs, ys, color=color, linewidth=2, label=name)
    line_ends.append([np.log10(ys[-1]), name, xs[-1]])
ax_grad.set_yscale('log')
ax_grad.margins(x=0.22)
lo, hi = np.log10(ax_grad.get_ylim())
line_ends.sort()
for prev, cur in zip(line_ends, line_ends[1:]):
    cur[0] = max(cur[0], prev[0] + 0.045 * (hi - lo))
for y, name, x in line_ends:
    ax_grad.annotate(name, (x, 10 ** y), xytext=(4, 0), textcoords='offset points', va='center', fontsize=8, color=INK)
ax_grad.set(title='Backprop: gradient size reaching each layer', xlabel='step', ylabel='gradient norm (log)')
ax_grad.legend(frameon=False, fontsize=8, title='input -> output', title_fontsize=8)

sample = out[:block_size]
if USE_ATTENTION:
    with torch.no_grad():
        model(torch.tensor([encode(sample)], dtype=torch.long, device=device))
labels = [{'\n': '\\n', ' ': '·'}.get(c, c) for c in sample]
cmap = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#5598e7', '#0d366b'])
for ax, (layer, head) in zip([ax_att1, ax_att2], PLOT_HEADS):
    if not USE_ATTENTION:
        ax.axis('off')
        ax.text(0.5, 0.5, 'Attention is off\n(turn on USE_ATTENTION)', ha='center', va='center', color=MUTED)
        continue
    att = model.blocks[layer - 1].sa.heads[head - 1].last_attention[0].cpu().numpy()
    ax.imshow(att, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.grid(False)
    ax.set(title=f'Attention: layer {layer}, head {head}', xlabel='looking at (key)', ylabel='current character (query)')

fig.tight_layout()
fig.savefig(os.path.join(HERE, 'training_plots.png'), dpi=120)
print(f"\nplots saved to {os.path.join(HERE, 'training_plots.png')}")
if SHOW_PLOTS:
    plt.show()
