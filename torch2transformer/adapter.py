# torch2transformer/adapter.py
import torch
import torch.nn as nn
from typing import Optional, Dict

class TorchAdapter(nn.Module):
    """
    Minimal interface for torch2transformer compatibility.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Must return:
          {
            "logits": Tensor,
            "loss": Tensor | None
          }
        """
        raise NotImplementedError
