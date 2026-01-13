# torch2transformer/config.py
from transformers import PretrainedConfig
from .version import __version__

class Torch2TransformerConfig(PretrainedConfig):
    model_type = "torch2transformer"

    def __init__(
        self,
        torch_model_kwargs=None,
        task_type="causal_lm",
        torch2transformer_version=__version__,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.torch_model_kwargs = torch_model_kwargs or {}
        self.task_type = task_type
        self.torch2transformer_version = torch2transformer_version

        # Generation-related flags
        self.is_decoder = True
        self.is_encoder_decoder = False
        self.use_cache = False

        # Optional but nice
        self.architectures = ["Torch2TransformerModel"]
