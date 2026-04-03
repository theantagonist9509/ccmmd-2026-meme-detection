# %%
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

# %%
val_data = mud.create_val_dataset(os.getenv("VAL_PATH"))
train_data = mud.create_train_dataset(os.getenv("TRAIN_PATH"))

# %%
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# %%
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

class TextClassifier(nn.Module):
    def __init__(self, model_name="answerdotai/ModernBERT-base", num_classes=2, dropout_rate=0.3):
        super(TextClassifier, self).__init__()
        
        # AutoModelForSequenceClassification automatically adds the classification head
        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            classifier_dropout=dropout_rate
        )

    def forward(self, input_ids, attention_mask):
        # The AutoModel returns a SequenceClassifierOutput object; we just need the logits
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

# %%

# --- NEW: Initialize tracking variables ---
best_f1 = 0.0
save_dir = "../weights/bert/best_model"
os.makedirs(save_dir, exist_ok=True)
# ------------------------------------------

# %%
model_name = "answerdotai/ModernBERT-base" 
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer

# Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TextClassifier(model_name=model_name).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4)

# Example Training Loop
epochs = 20

for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        
        texts = batch["transcription"]
        labels = batch["indian_label"].to(device) 
        
        encoded_inputs = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        ).to(device)
        
        logits = model(
            input_ids=encoded_inputs['input_ids'], 
            attention_mask=encoded_inputs['attention_mask']
        )
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)

    # --- Validation Phase ---
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{epochs}"):
            texts = batch["transcription"]
            labels = batch["indian_label"].to(device)
            
            encoded_inputs = tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=256, 
                return_tensors="pt"
            ).to(device)
            
            logits = model(
                input_ids=encoded_inputs['input_ids'], 
                attention_mask=encoded_inputs['attention_mask']
            )
            
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            # Convert logits to predicted class (0 or 1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    
    # --- Calculate Macro Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Val Accuracy:  {acc:.4f}")
    print(f"Val Precision: {prec:.4f}")
    print(f"Val Recall:    {rec:.4f}")
    print(f"Val F1 Score:  {f1:.4f}\n")
    print("-" * 50)

    # --- NEW: Check and Save Best Model ---
    if f1 > best_f1:
        print(f"--> F1 improved from {best_f1:.4f} to {f1:.4f}. Saving model...")
        best_f1 = f1
        
        # Save the full PyTorch state dict
        torch.save(model.state_dict(), os.path.join(save_dir, "best_classifier.pth"))
        
        # Optional but highly recommended: save the HuggingFace components for easy loading later
        model.encoder.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
    # --------------------------------------


