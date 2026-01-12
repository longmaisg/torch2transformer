
# Step 1: Prepare toy data
posts = [
    "I love sunny days",
    "Reading books is fun",
    "Cooking pasta tonight",
    "Going for a walk",
    "Python is great"
]

# Flatten into one string for character-level LM
text = " ".join(posts)
chars = sorted(list(set(text)))  # unique characters
vocab_size = len(chars)
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

# Convert text to sequence of IDs
data = [char2id[c] for c in text]

# Step 2: Create sequences for training
seq_len = 10
X, Y = [], []

for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    Y.append(data[i+1:i+seq_len+1])  # next-char prediction

import torch

X = torch.tensor(X)  # [num_samples, seq_len]
Y = torch.tensor(Y)  # [num_samples, seq_len]
print(X.shape, Y.shape)

# Step 3: Build a tiny PyTorch model
import torch.nn.functional as F
import torch.nn as nn
from torch2transformer import TorchAdapter

class TinyCharModel(TorchAdapter):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)       # [batch, seq, hidden]
        x, _ = self.rnn(x)
        logits = self.fc(x)             # [batch, seq, vocab_size]

        loss = None
        if labels is not None:
            # reshape for cross-entropy: [batch*seq_len, vocab]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1))
        return {"logits": logits, "loss": loss}


# # Step 4: Wrap it into a Transformers model
# from transformers import PreTrainedModel, PretrainedConfig

# class TinyConfig(PretrainedConfig):
#     model_type = "tiny_char_model"
#     def __init__(self, vocab_size=vocab_size, hidden_size=32, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab_size = vocab_size
#         self.hidden_size = hidden_size

# class TinyTransformersModel(PreTrainedModel):
#     config_class = TinyConfig
    
#     def __init__(self, config):
#         super().__init__(config)
#         self.model = TinyCharModel(config.vocab_size, config.hidden_size)
#         self.post_init()
    
#     def forward(self, input_ids=None, labels=None, **kwargs):
#         return self.model(input_ids, labels=labels)

# Step 5: Prepare Dataset for Trainer
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "labels": self.Y[idx]
        }

train_dataset = CharDataset(X, Y)

# Step 6: Train with Trainer
from transformers import Trainer, TrainingArguments

# model = TinyTransformersModel(TinyConfig())

# Wrap the model for HF Trainer
from torch2transformer import TorchAdapter, wrap_model, load_model
model = wrap_model(
    torch_model_cls=TinyCharModel,
    torch_model_kwargs={"vocab_size": 100, "hidden_size": 32},
    task_type="causal_lm"
)

print(model)


training_args = TrainingArguments(
    output_dir="/Users/longmai/projects/TinyCharModel/tiny_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"  # disable WandB / logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

trainer.train()

model.save_pretrained("/Users/longmai/projects/torch2transformer/tiny_ckpt")

model = load_model("/Users/longmai/projects/torch2transformer/tiny_ckpt", torch_model_cls=TinyCharModel)


# Step 7: Generate text
# pick a seed
seed_text = "I love "
input_ids = torch.tensor([[char2id[c] for c in seed_text]])

device = "cpu"
model.to(device)
input_ids = input_ids.to(device)
# labels = labels.to(device)  # if labels exist


model.eval()
with torch.no_grad():
    logits = model(input_ids)["logits"]
    pred_ids = torch.argmax(logits, dim=-1)[0]

pred_text = "".join([id2char[i.item()] for i in pred_ids])
print("Seed:", seed_text)
print("Generated:", pred_text)
