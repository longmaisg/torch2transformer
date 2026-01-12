import torch
import torch.nn as nn
import torch.nn.functional as F
from torch2transformer import TorchAdapter, wrap_model

# 1. Define a tiny char model
class TinyCharModel(TorchAdapter):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        x, _ = self.rnn(x)
        logits = self.fc(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return {"logits": logits, "loss": loss}

# 2. Wrap the model for HF Trainer
model = wrap_model(
    torch_model_cls=TinyCharModel,
    torch_model_kwargs={"vocab_size": 100, "hidden_size": 32},
    task_type="causal_lm"
)

print(model)
