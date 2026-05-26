# LLM From Scratch

A PyTorch notebook that builds a character-level language model step by step.

The main file is `pipeLine.ipynb`.

## Model Development

| Step | Model / Idea | Improvement |
| --- | --- | --- |
| 1 | **Bigram baseline** | Each character predicts the next character directly. Simple baseline, but no real context. |
| 2 | **Context averaging** | Each token uses the average of previous tokens, introducing past context. |
| 3 | **Causal masking** | A lower-triangular matrix prevents tokens from seeing future tokens. |
| 4 | **Softmax attention** | Masked scores become attention probabilities instead of fixed averages. |
| 5 | **Single attention head** | Learned `key`, `query`, and `value` projections let tokens choose relevant past tokens. |
| 6 | **Token + positional embeddings** | The model represents both character identity and position in the sequence. |
| 7 | **Multi-head attention** | Multiple heads learn different relationships in parallel. |
| 8 | **Feed-forward network** | Adds nonlinear processing after attention. |
| 9 | **Transformer blocks** | Stacks attention and feed-forward layers with residual connections, layer norm, and dropout. |


## Final Model Flow

```text
token ids
-> token embeddings + positional embeddings
-> stacked Transformer blocks
-> final layer norm
-> language-model head
-> next-character logits
```

## Files

- `pipeLine.ipynb` - full model-development notebook.
- `input.txt` - training text.
- `generated_text.txt` - generated sample output.
- `bigram_model.pth` - saved checkpoint.

## Run

Open `pipeLine.ipynb` and run the cells from top to bottom. CUDA is used automatically if available.

## Notes

This exploration is based on Andrej Karpathy's `nanoGPT` / "Let's build GPT from scratch" material.
