# %% 
import os
import torch
from PIL import Image
from transformers import set_seed, Qwen3VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from transformers import Trainer, TrainingArguments
from peft import get_peft_model

import os
import sys
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

load_dotenv()
sys.path.append(os.getenv("ROOT_PATH"))
import miso_utils.datasets as mud

# ---------------------------------------------------------------------------
# 1. Environment & Setup
# ---------------------------------------------------------------------------
set_seed(42)

# A100/H100 optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

# ---------------------------------------------------------------------------
# 2. Dataset Initialization
# ---------------------------------------------------------------------------
# Pass an identity function to keep images as PIL.Image objects instead of tensors
identity_transform = lambda x: x

train_ds = mud.create_train_dataset(os.getenv("TRAIN_PATH"),image_transform = identity_transform)

# %%
# ---------------------------------------------------------------------------
# 3. Model & Processor
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-VL-4B-Thinking"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    dtype=torch.bfloat16, 
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# %%
# ---------------------------------------------------------------------------
# 4. Collate Function (Dynamic Formatting)
# ---------------------------------------------------------------------------
MAX_LEN = 512 
MAX_IMAGE_SIDE = 640 
MAX_IMAGE_PIXELS = 640 * 640

def _resize_pil(pil: Image.Image, max_side: int = MAX_IMAGE_SIDE, max_pixels: int = MAX_IMAGE_PIXELS) -> Image.Image:
    w, h = pil.size
    scale_side = min(1.0, max_side / float(max(w, h)))
    scale_area = (max_pixels / float(w * h)) ** 0.5 if (w * h) > max_pixels else 1.0
    scale = min(scale_side, scale_area)

    if scale < 1.0:
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        pil = pil.resize((nw, nh), resample=Image.BICUBIC)
    return pil

def collate_fn(batch):
    full_texts = []
    prompt_texts = []
    images = []

    for ex in batch:
        # 1. Extract raw data from MisoDataset output
        pil_img = _resize_pil(ex["img"]) 
        transcription = ex["transcription"]
        target_label = str(ex["indian_label"]) 

        # 2. Build the instruction prompt
        prompt = (
            f"Transcription: {transcription}\n"
            "Analyze the image and the transcription. Determine if the content is misogynistic.\n"
            "Output exactly '1' for misogyny or '0' for non-misogyny. Output nothing else."
        )

        # 3. Format into chat dictionary
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_label}],
            },
        ]

        # 4. Apply chat template
        full_texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
        prompt_texts.append(processor.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True))
        images.append(pil_img)

    # Tokenize everything together
    enc = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )

    input_ids = enc["input_ids"]
    pad_id = processor.tokenizer.pad_token_id

    # Compute prompt lengths to mask them in the loss function
    prompt_ids = processor.tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=False,
    )["input_ids"]

    prompt_lens = (prompt_ids != pad_id).sum(dim=1)
    labels = input_ids.clone()
    bs, seqlen = labels.shape

    # Mask prompt and padding tokens
    for i in range(bs):
        pl = int(prompt_lens[i].item())
        pl = min(pl, seqlen)
        labels[i, :pl] = -100

    labels[labels == pad_id] = -100
    enc["labels"] = labels
    return enc

# ---------------------------------------------------------------------------
# 7. Inference Test on a Dataset Sample
# ---------------------------------------------------------------------------
# %%
def run_inference(model_, dataset_item, max_new_tokens=10):
    pil_img = _resize_pil(dataset_item["img"])
    prompt = (
        f"Transcription: {dataset_item['transcription']}\n"
        "Analyze the image and the transcription. Determine if the content is misogynistic.\n"
        "Output '1' for misogyny or '0' for non-misogyny"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_.device)

    with torch.inference_mode():
        out = model_.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()

test_ex = train_ds[0]
print("\n--- FINETUNED OUTPUT ---")
print("Prediction:", run_inference(model, test_ex))
print("Target:", test_ex["indian_label"])

# %%

val_ds = mud.create_val_dataset(os.getenv("VAL_PATH"),image_transform=identity_transform)

def run_inference_choice(model_, dataset_item, max_new_tokens=400):
    pil_img = _resize_pil(dataset_item["img"])
    # prompt = (
    #     f"Transcription: {dataset_item['transcription']}\n"
    #     "This image contains non-misogynistic content.\n"
    #     "Show a step-by-step chain of thought reaching that conclusion"
    # )

    prompt = (
        f"Transcription: {dataset_item['transcription']}\n"
        "Analyze the image and the transcription. Determine if the content is misogynistic.\n"
        "Output '1' for misogyny or '0' for non-misogyny"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_.device)

    with torch.inference_mode():
        out = model_.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(gen, skip_special_tokens=True).strip()

# %%

test_ex = val_ds[0]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])

test_ex = val_ds[1]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])

test_ex = val_ds[2]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])


test_ex = val_ds[3]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])


test_ex = val_ds[4]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])


