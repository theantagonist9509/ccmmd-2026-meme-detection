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
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

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
# 5. Training Setup
# ---------------------------------------------------------------------------
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

# 1. Apply LoRA to the model manually
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 2. Use TrainingArguments instead of SFTConfig
args = TrainingArguments(
    output_dir="qwen3vl-miso-lora",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    learning_rate=1e-4,             
    warmup_steps=10,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,
    fp16=False,
    lr_scheduler_type="cosine",
    logging_steps=10,
    report_to="none",
    remove_unused_columns=False, # Essential: prevents Trainer from dropping dataset keys
)

# %%

# 3. Use standard Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    data_collator=collate_fn,
)

trainer.train()

# ---------------------------------------------------------------------------
# 6. Save & Push
# ---------------------------------------------------------------------------
out_dir = trainer.args.output_dir
trainer.save_model(out_dir)
processor.save_pretrained(out_dir)

# ---------------------------------------------------------------------------
# 7. Inference Test on a Dataset Sample
# ---------------------------------------------------------------------------
# %%
def run_inference(model_, dataset_item, max_new_tokens=10):
    pil_img = _resize_pil(dataset_item["img"])
    prompt = (
        f"Transcription: {dataset_item['transcription']}\n"
        "Analyze the image and the transcription. Determine if the content is misogynistic.\n"
        "Output exactly '1' for misogyny or '0' for non-misogyny. Output nothing else."
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

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

# Paths based on your training script
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LORA_DIR = "qwen3vl-miso-lora"

# 1. Load the processor (saved during your script)
processor = AutoProcessor.from_pretrained(LORA_DIR)

# 2. Load the original base model
base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)

# 3. Load and attach the trained LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_DIR)

# Optional: Merge the LoRA weights into the base model for slightly faster inference
# model = model.merge_and_unload()

model.eval()

# %%

val_ds = mud.create_val_dataset(os.getenv("VAL_PATH"),image_transform=identity_transform)

preds = []
labels = []
from tqdm import tqdm
for i in tqdm(range(len(val_ds))):
    pred = run_inference(model,val_ds[i])
    if "0" <= pred[0] <= '1':
        preds.append(int(pred[0]))
    else: preds.append(2)
    labels.append(val_ds[i]["indian_label"])

import matplotlib.pyplot as plt
import math

# 1. Find indices of all False Negatives (True = 1, Pred = 0)
fn_indices = [i for i in range(len(labels)) if labels[i] == 1 and preds[i] == 0]

print(f"Total False Negatives found: {len(fn_indices)}")

if fn_indices:
    # 2. Set up a dynamic grid (max 5 columns)
    cols = min(5, len(fn_indices))
    rows = math.ceil(len(fn_indices) / cols)
    
    # Scale figure size based on rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    
    # Flatten axes array for easy iteration (handles single-row edge case)
    if len(fn_indices) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    # 3. Plot each image
    for ax_idx, ds_idx in enumerate(fn_indices):
        ax = axes[ax_idx]
        
        # Extract the PIL image (adjust "img" to "image" if your val_ds uses that key)
        img = val_ds[ds_idx].get("img") or val_ds[ds_idx].get("image")
        
        ax.imshow(img)
        ax.set_title(f"Idx: {ds_idx}\nPred: 0 | True: 1", fontsize=10)
        ax.axis("off")
        
    # 4. Hide any empty subplots in the grid
    for i in range(len(fn_indices), len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.show()

fp_indices = [i for i in range(len(labels)) if labels[i] == 0 and preds[i] == 1]

print(f"Total False Positives found: {len(fp_indices)}")

if fp_indices:
    # 2. Set up a dynamic grid (max 5 columns)
    cols = min(5, len(fp_indices))
    rows = math.ceil(len(fp_indices) / cols)
    
    # Scale figure size based on rows
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5 * rows))
    
    # Flatten axes array for easy iteration (handles single-row edge case)
    if len(fp_indices) > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
        
    # 3. Plot each image
    for ax_idx, ds_idx in enumerate(fp_indices):
        ax = axes[ax_idx]
        
        # Extract the PIL image (adjust "img" to "image" if your val_ds uses that key)
        img = val_ds[ds_idx].get("img") or val_ds[ds_idx].get("image")
        
        ax.imshow(img)
        ax.set_title(f"Idx: {ds_idx}\nPred: 1 | True: 0", fontsize=10)
        ax.axis("off")
        
    # 4. Hide any empty subplots in the grid
    for i in range(len(fp_indices), len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout()
    plt.show()



from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate Macro F1 Score
macro_f1 = f1_score(labels, preds, labels=[0, 1], average="macro")
print(f"Macro F1 Score: {macro_f1:.4f}")

# Generate and plot the Confusion Matrix
cm = confusion_matrix(labels, preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0 (Non-Miso)", "1 (Miso)"])

# Plot formatting
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Validation Confusion Matrix")
plt.show()

# %%

# preds = []
# labels = []
# from tqdm import tqdm
# for i in tqdm(range(9000)):
#     pred = run_inference(model,train_ds[i])
#     if "0" <= pred[0] <= '1':
#         preds.append(int(pred[0]))
#     else: preds.append(2)
#     labels.append(train_ds[i]["indian_label"])

# from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Calculate Macro F1 Score
# macro_f1 = f1_score(labels, preds, labels=[0, 1], average="macro")
# print(f"Macro F1 Score: {macro_f1:.4f}")

# # Generate and plot the Confusion Matrix
# cm = confusion_matrix(labels, preds, labels=[0, 1])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0 (Non-Miso)", "1 (Miso)"])

# # Plot formatting
# fig, ax = plt.subplots(figsize=(6, 6))
# disp.plot(ax=ax, cmap="Blues", values_format="d")
# plt.title("Validation Confusion Matrix")
# plt.show()

# %%


model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID, 
    dtype=torch.bfloat16, 
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

def run_inference_choice(model_, dataset_item, max_new_tokens=400):
    pil_img = _resize_pil(dataset_item["img"])
    prompt = (
        f"Transcription: {dataset_item['transcription']}\n"
        "This image contains non-misogynistic content.\n"
        "Show a step-by-step chain of thought reaching that conclusion"
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

test_ex = val_ds[fp_indices[4]]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])

test_ex = val_ds[fp_indices[5]]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])

test_ex = val_ds[fp_indices[6]]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])


test_ex = val_ds[fp_indices[7]]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])


test_ex = val_ds[fp_indices[8]]
print("\n--- OUTPUT ---")
print("Prediction:", run_inference_choice(model, test_ex))
print("Target:", test_ex["indian_label"])


