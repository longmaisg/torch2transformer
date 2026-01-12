import torch
from torch2transformer import TorchAdapter, wrap_model

class DummyModel(TorchAdapter):
    def forward(self, input_ids, labels=None):
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 10)
        loss = None
        if labels is not None:
            loss = torch.mean(logits)
        return {"logits": logits, "loss": loss}

def test_wrap_model():
    model = wrap_model(
        torch_model_cls=DummyModel,
        torch_model_kwargs={},
        task_type="causal_lm"
    )
    input_ids = torch.randint(0, 10, (2, 5))
    out = model(input_ids=input_ids)
    assert "logits" in out
    assert out["logits"].shape == (2, 5, 10)

if __name__ == "__main__":
    test_wrap_model()
    print("All tests passed.")
