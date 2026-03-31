# %% [markdown]
# # BERT Dataset Generation
# This notebook generates BERT embeddings for the MisoDataset.
# %%
import os
import sys
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getenv("ROOT_PATH"))
import miso_utils.datasets as mud

# %%
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Add the root directory to sys.path to ensure miso_utils can be imported
sys.path.append(str(Path(__file__).parent.parent))

from miso_utils.datasets import create_train_dataset, EmbeddedMisoDataset,create_val_dataset

# %%
# Global Configurations
MODEL_NAME = "clip_image"
DATASET_PATH = os.getenv("VAL_PATH")
SAVE_PATH = f"./datasets/{MODEL_NAME}/val.pt"
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH
# %%
import torch
from PIL import Image
import open_clip
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
clip_model = clip_model.to(device) 
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')


class ClipWrapperText(nn.Module):

    def __init__(self):
        super().__init__()
        self.tokenizer = clip_tokenizer
        self.clip_model = clip_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, batch):
        text_tokens = clip_tokenizer(batch['transcription']).to(device)
        txt_emb = clip_model.encode_text(text_tokens)
        txt_emb /= txt_emb.norm(dim=-1, keepdim=True)
        return txt_emb

class ClipWrapperImage(nn.Module):

    def __init__(self):
        super().__init__()
        self.tokenizer = clip_tokenizer
        self.clip_model = clip_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, batch):
        images = batch['img'].to(device)
        img_emb = clip_model.encode_image(images)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        return img_emb


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
model = ClipWrapperImage()

# %%
miso_train = create_val_dataset(path=DATASET_PATH, mode="image")

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

# %%
