# torch2transformer/model.py
from transformers import PreTrainedModel
from .config import Torch2TransformerConfig

class Torch2TransformerModel(PreTrainedModel):
    config_class = Torch2TransformerConfig

    def __init__(self, config: Torch2TransformerConfig):
        super().__init__(config)

        if config.torch_model_cls is None:
            raise ValueError("torch_model_cls must be provided")

        self.model = config.torch_model_cls(
            **config.torch_model_kwargs
        )

        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            **kwargs
        )
