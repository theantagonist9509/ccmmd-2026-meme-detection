# %% [markdown]
# # BERT Dataset Generation
# This notebook generates BERT embeddings for the MisoDataset.

# %%
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Add the root directory to sys.path to ensure miso_utils can be imported
sys.path.append(str(Path(__file__).parent.parent))

from miso_utils.datasets import create_train_dataset, EmbeddedMisoDataset

# %%
# Global Configurations
MODEL_NAME = "bert-base-uncased"
DATASET_PATH = "./datasets/original/train"
SAVE_PATH = f"./datasets/{MODEL_NAME}/train.pt"
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
class BertWrapper(nn.Module):
    """
    A wrapper around a pretrained BERT model that accepts a dictionary batch
    (as produced by MisoDataset) and returns the encoded embeddings.
    """
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, batch):
        texts = batch["transcription"]
        device = next(self.encoder.parameters()).device

        # tokenization works on lists of strings which is what the dataloader passes for text
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.encoder(**inputs)

        # Return the [CLS] token representation / pooler output
        return outputs.pooler_output

# %%
model = BertWrapper(model_name=MODEL_NAME)

# %%
miso_train = create_train_dataset(path=DATASET_PATH, mode="text")

# %%
embedded_dataset = EmbeddedMisoDataset(
    miso_dataset=miso_train,
    model=model,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

# %%
save_dir = Path(SAVE_PATH).parent
save_dir.mkdir(parents=True, exist_ok=True)
torch.save(embedded_dataset, SAVE_PATH)
