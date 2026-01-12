# torch2transformer/config.py
from transformers import PretrainedConfig

class Torch2TransformerConfig(PretrainedConfig):
    model_type = "torch2transformer"

    def __init__(
        self,
        torch_model_kwargs=None,
        task_type="causal_lm",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.torch_model_kwargs = torch_model_kwargs or {}
        self.task_type = task_type
