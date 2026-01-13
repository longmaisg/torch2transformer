# torch2transformer

**torch2transformer** lets you wrap plain PyTorch models so they work
seamlessly with the Hugging Face Transformers ecosystem. Important note: The original PyTorch model class must be available at run time and load time.

## Features
- Use `Trainer` with any PyTorch model
- Save / load via `save_pretrained`
- Minimal adapter interface
- No custom training loops

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
