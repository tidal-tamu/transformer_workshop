## Transformers - Mini-GPT Workshop

Build a tiny GPT, one character at a time. You'll start with a bigram model (each character only looks at the one before it) and add the pieces of a transformer one by one: **self-attention**, **feed-forward layers**, and **residual connections**. You'll see the generated text go from gibberish to something that reads like a Wikipedia article.

This is a **decoder-only** transformer (the same kind of model as GPT). A causal mask stops each character from looking at future characters.

### What's in this folder

| File | What it is |
|------|------------|
| `starter_code.py` | **The file you work in.** Fill in the `___` blanks, then change the settings at the top. |
| `completed_code/mini_gpt.py` | **The answer key and tuned reference.** The blanks filled in, plus weight decay, learning-rate decay and early stopping. |
| `France_Wikipedia_Clean_Text.txt` | The training text. |
| `requirements.txt` | The Python packages to install. You get it when you clone the repo, so there's nothing to download separately. |

### Setup

0. **Clone the repository**:

```
git clone https://github.com/tidal-tamu/transformer_workshop.git
cd transformer_workshop
```

1. **Create and activate a virtual environment**:

-   macOS/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

-   Windows (Command Prompt):

```
python -m venv venv
venv\Scripts\activate
```

2. **Install dependencies** (installs everything listed in `requirements.txt`: torch, numpy and matplotlib):

```
pip install -r requirements.txt
```

3. **Deactivate the virtual environment** when done:

JUST TO CLARIFY, THIS IS DONE AT THE END, NOT THE BEGINNING!

```
deactivate
```

### Running

```
python starter_code.py
```

Training takes about 1-3 minutes on a laptop CPU. If yours is slow, lower `max_iters`. When training finishes, the script prints a text sample, then opens a window of plots (also saved as `training_plots.png`). **Close the plot window to end the program.**

### Reading the Plots

| Plot | What it shows |
|------|---------------|
| **Gradient descent** | Loss over training. The faint line is every noisy mini-batch step. If train keeps falling while val levels off or rises, the model is memorizing. |
| **Backprop** | How big the gradient reaching each layer is (light = input, dark = output). Without residuals, the gradients reaching the first layers collapse early in training (vanishing gradients) while the loss stalls. |
| **Attention** | Which earlier characters (columns) each character (row) looks at. The empty upper triangle is the causal mask. |
| **Learning rate** | Step size over training. Flat in the starter; `mini_gpt.py` decays it toward 0. |

> Want the answers? `completed_code/mini_gpt.py` shows what tuning gets you on top of the basic model.
