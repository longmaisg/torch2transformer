# torch2transformer/factory.py
from .config import Torch2TransformerConfig
from .model import Torch2TransformerModel

def wrap_model(
    torch_model_cls,
    torch_model_kwargs,
    task_type="causal_lm",
    **hf_kwargs
):
    """
    Wrap a plain PyTorch model for Hugging Face Trainer.
    """
    config = Torch2TransformerConfig(
        torch_model_kwargs=torch_model_kwargs,
        task_type=task_type,
        **hf_kwargs
    )

    model = Torch2TransformerModel(
        config=config,
        torch_model_cls=torch_model_cls
    )
    return model

# factory.py
def load_model(path, torch_model_cls, **kwargs):
    return Torch2TransformerModel.from_pretrained(path, torch_model_cls=torch_model_cls, **kwargs)

