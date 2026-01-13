# torch2transformer/model.py
from transformers import PreTrainedModel
from .config import Torch2TransformerConfig
from .version import __version__
import torch
import os
import warnings
from packaging import version

class Torch2TransformerModel(PreTrainedModel):
    config_class = Torch2TransformerConfig

    def __init__(self, config, torch_model_cls=None):
        super().__init__(config)

        if torch_model_cls is None:
            raise ValueError(
                "torch_model_cls must be provided at runtime. "
                "It cannot be stored in the config."
            )

        self.torch_model = torch_model_cls(**config.torch_model_kwargs)
        self.post_init()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.torch_model(
            input_ids=input_ids,
            labels=labels,
            **kwargs
        )
    
    @classmethod
    def from_pretrained(cls, path, torch_model_cls, **kwargs):
        from .config import Torch2TransformerConfig

        # load config
        config = Torch2TransformerConfig.from_pretrained(path, **kwargs)

        # check version compatibility
        saved_version = getattr(config, "torch2transformer_version", None)
        if saved_version is not None:
            if version.parse(saved_version) > version.parse(__version__):
                warnings.warn(
                    f"Checkpoint was created with torch2transformer {saved_version}, "
                    f"but you are running {__version__}. "
                    "Unexpected behavior may occur.",
                    UserWarning,
                )
        else:
            warnings.warn(
                "Checkpoint was created before torch2transformer versioning existed. "
                "Assuming compatibility.",
                UserWarning,
            )

        # create model instance
        model = cls(config=config, torch_model_cls=torch_model_cls)

        # find weights file
        safetensors_path = os.path.join(path, "model.safetensors")
        pytorch_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No weights found at {safetensors_path} or {pytorch_path}"
            )

        model.load_state_dict(state_dict)
        return model
            
