# torch2transformer

**torch2transformer** lets you wrap plain PyTorch models so they work
seamlessly with the Hugging Face Transformers ecosystem.  

**Important:** The original PyTorch model class must be available at **run time** and **load time**.

---

## Features

- Wrap any PyTorch model for Hugging Face `Trainer`
- Save / load via `save_pretrained` / `from_pretrained`
- Supports Hugging Face `.generate()` for autoregressive text generation
- Minimal interface required: `forward(input_ids, labels=None)` returning `{"logits": ..., "loss": ...}`
- No custom training loops needed

---

## Installation

```bash
pip install torch2transformer
or,
uv pip install torch2transformer
```

## Example

```python
from torch2transformer import TorchAdapter, wrap_model, load_model

# wrap Pytorch model as a Transformer model
model = wrap_model(
    torch_model_cls=TinyCharModel,
    torch_model_kwargs={"vocab_size": 100, "hidden_size": 32},
    task_type="causal_lm"
)
# then can be used with Trainer()

# save model
model.save_pretrained("./tiny_ckpt")

# load model
model = load_model("./tiny_ckpt", torch_model_cls=TinyCharModel)

# example input
seed_text = "I love "
input_ids = torch.tensor([[char2id[c] for c in seed_text]])

# generate next characters
output_ids = model.generate(input_ids, max_new_tokens=20)
pred_text = "".join([id2char[i] for i in output_ids[0].tolist()])
print(pred_text)