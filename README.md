## Transformers - Mini-GPT Shakespeare Workshop

Build a tiny GPT that writes Shakespeare, one character at a time. You'll start with a bigram model (each character only looks at the one before it) and add the pieces of a transformer one by one: **self-attention**, **feed-forward layers**, and **residual connections**. You'll see the generated text go from gibberish to something that looks like a play.

This is a **decoder-only** transformer (the same kind of model as GPT). A causal mask stops each character from looking at future characters.

### Setup

0. **Clone the repository**:

```
git clone https://github.com/tidal-tamu/transformers-workshop.git
cd transformers-workshop
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

2. **Install dependencies**:

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

Training takes about 1-3 minutes on a laptop CPU. If yours is slow, lower `max_iters`.

### Workshop Steps

Fill in the `___` blanks in `starter_code.py`, then turn on one flag at a time and rerun:

| Step | Flags on | What you should see |
|------|----------|---------------------|
| 1 | none (bigram) | Shakespeare-shaped gibberish |
| 2 | `USE_ATTENTION` | Lower loss, word-ish chunks |
| 3 | `+ USE_FFWD` | ...loss goes **up**?! |
| 4 | `+ USE_RESIDUAL` | Character names and real words |

Things to try afterwards:
-   Change `temperature` to 0.3, then 1.5.
-   Set `block_size` to 2 or to 64.
-   Delete the causal mask line and see what happens.

> Stuck? See `completed_code/mini_gpt.py`.

### Deploying (organizers only)

Push this folder to a new repo under the tidal-tamu GitHub org so attendees can clone it:

```
git init
git add .
git commit -m "Transformers workshop"
git branch -M main
git remote add origin https://github.com/tidal-tamu/transformers-workshop.git
git push -u origin main
```

Before pushing, put the blanks back into `starter_code.py`. Attendees should get a starter with blanks, not the answers. The answers are in `completed_code/mini_gpt.py`.
