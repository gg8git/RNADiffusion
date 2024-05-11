# Requirements:

Use the latest pytorch if possible (2.3.0)

```bash
pip install torch torchvision torchaudio
```

```bash
pip install \
    einops \
    fire \
    hnn-utils \
    lightning \
    rich \
    schedulefree \
    wandb
```

If your GPUs are new enough (ie. Quadro A5000, A6000, A6000 Ada, 3000 series, etc.), you can use flash attention which will decrease memory usage and increase speed

```bash
pip install flash-attn --no-build-isolation
```

# Data

Need to set up the logic under `data/vae_datamodule.py` to load the data. The model expects a tensor of shape `[B, Seq_Len]` of integers representing tokens.
Add whatever tokens are needed to `data/vocab.json`. I imagine it will look like
```json
{
    "[START]": 0,
    "[STOP]": 1,
    "[PAD]": 2,
    "[UNK]": 3,
    "A": 4,
    "C": 5,
    "G": 6,
    "T": 7,
}