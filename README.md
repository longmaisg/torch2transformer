# torch2transformer

**torch2transformer** lets you wrap plain PyTorch models so they work
seamlessly with the Hugging Face Transformers ecosystem.

## Features
- Use `Trainer` with any PyTorch model
- Save / load via `save_pretrained`
- Minimal adapter interface
- No custom training loops

## Example

```python
from torch2transformer import wrap_model

model = wrap_model(
    torch_model_cls=MyTorchModel,
    torch_model_kwargs={"vocab_size": 100},
)

trainer = Trainer(model=model, ...)
trainer.train()
