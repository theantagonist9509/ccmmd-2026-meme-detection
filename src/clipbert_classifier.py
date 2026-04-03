# %%
import os
import sys
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import open_clip
load_dotenv()
sys.path.append(os.getenv("ROOT_PATH"))
import miso_utils.datasets as mud
from transformers import AutoModel, AutoTokenizer

# %%
device = "cuda"

class ANNClassificationHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=128, num_classes=2, dropout_rate=0.4):
        super().__init__()
        
        self.net = nn.Sequential(
            # First Layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Second Layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer (Logits)
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        # x shape should be [batch_size, 1280]
        logits = self.net(x)
        return logits


class ClipBertModel(nn.Module):
    def __init__(self):
        super().__init__()

        # init CliP
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', precision='fp32')
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

        # init Bert
        MODEL_NAME = "bert-base-uncased"
        self.bert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.bert_encoder = AutoModel.from_pretrained(MODEL_NAME)

        # init head
        self.classifier = ANNClassificationHead(
            input_dim = 1280, 
            num_classes = 2 
        ).to(device)

    def clip_embedding(self,batch):
        images = batch['img'].to(device)
        img_emb = self.clip_model.encode_image(images)
        img_emb /= img_emb.norm(dim=-1, keepdim=True)
        return img_emb
    
    def bert_embedding(self,batch):
        texts = batch["transcription"]
        inputs = self.bert_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        outputs = self.bert_encoder(**inputs)
        return outputs.pooler_output
    
    def forward(self, batch):
        # Extract individual embeddings
        img_emb = self.clip_embedding(batch)
        text_emb = self.bert_embedding(batch)
        
        # Concatenate along the feature dimension (batch_size, clip_dim + bert_dim)
        # ViT-B-32 is 512, BERT-base is 768. Output shape: [batch_size, 1280]
        concat_emb = torch.cat((img_emb, text_emb), dim=1)

        logits = self.classifier(concat_emb)
        return logits

    def freeze_clip(self):
        """Sets requires_grad to False for all CLIP parameters."""
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
    def freeze_bert(self):
        """Sets requires_grad to False for all BERT parameters."""
        for param in self.bert_encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Utility method to unfreeze both models for joint fine-tuning stages."""
        for param in self.parameters():
            param.requires_grad = True
    
    def unfreeze_top_layers(self, num_bert_layers=0, num_clip_layers=0):
        """
        Unfreezes only the top N layers of BERT and CLIP, leaving the rest frozen.
        Call this instead of unfreeze_all() at the start of Phase 2.
        """
        # 1. Start by freezing everything to reset states
        self.freeze_bert()
        self.freeze_clip()
        
        # 2. Unfreeze top N layers of BERT
        for i in range(12 - num_bert_layers, 12):
            for param in self.bert_encoder.encoder.layer[i].parameters():
                param.requires_grad = True
                
        # Unfreeze BERT pooler (CRITICAL because you use outputs.pooler_output)
        for param in self.bert_encoder.pooler.parameters():
            param.requires_grad = True
            
        # 3. Unfreeze top N layers of CLIP Vision Transformer
        for i in range(12 - num_clip_layers, 12):
            for param in self.clip_model.visual.transformer.resblocks[i].parameters():
                param.requires_grad = True
                
        # Unfreeze CLIP's final layer norm and projection parameters
        for param in self.clip_model.visual.ln_post.parameters():
            param.requires_grad = True
            
        if self.clip_model.visual.proj is not None:
            self.clip_model.visual.proj.requires_grad = True

# %%

train_dataset = mud.create_train_dataset(os.getenv("TRAIN_PATH"))
val_dataset = mud.create_val_dataset(os.getenv("VAL_PATH"))

train_loader = DataLoader(train_dataset, batch_size= 8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size= 8, shuffle=False)

# %%

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score

def print_model_parameters(model):
    print(f"{'Component':<25} | {'Total Params':<15} | {'Trainable':<15} | {'Frozen':<15}")
    print("-" * 75)
    
    total_params = 0
    total_trainable = 0
    
    # Iterate through the main sub-modules
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen = params - trainable
        
        print(f"{name:<25} | {params:<15,} | {trainable:<15,} | {frozen:<15,}")
        
        total_params += params
        total_trainable += trainable

    print("-" * 75)
    print(f"{'TOTAL':<25} | {total_params:<15,} | {total_trainable:<15,} | {total_params - total_trainable:<15,}\n")

def train_multimodal_model(model, train_loader, val_loader, device="cuda", save_dir="../weights/clipbert/"):
    print_model_parameters(model)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_clipbert.pth")
    
    # Initialize Loss
    criterion = nn.CrossEntropyLoss()
    
    # ==========================================
    # PHASE 1: WARMUP HEAD (10 Epochs)
    # ==========================================
    print("Starting Phase 1: Warming up the classification head...")
    # Freeze both backbones so only the head trains
    model.freeze_clip()
    model.freeze_bert()
    print_model_parameters(model)
    
    # Optimizer specifically for the head (Standard LR)
    optimizer_phase1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    for epoch in range(3):
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch+1}/3"):
            # Assuming your dataset yields the label under 'indian_label' based on prior code
            labels = batch["indian_label"].to(device) 
            
            optimizer_phase1.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer_phase1.step()
            running_loss += loss.item()
            
        print(f"Phase 1 - Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f}")
        
    # ==========================================
    # PHASE 2: FINE-TUNE BERT + HEAD (20 Epochs)
    # ==========================================
    print("\nStarting Phase 2: Fine-tuning BERT and Head...")
    
    # Unfreeze everything, then freeze CLIP again (leaving BERT and Head active)
    model.unfreeze_top_layers()
    model.freeze_clip()
    print_model_parameters(model)
    
    # Optimizer for fine-tuning (Micro LR to protect BERT weights)
    optimizer_phase2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    
    best_f1 = 0.0
    
    for epoch in range(20):
        # --- Training ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Phase 2 - Epoch {epoch+1}/20 [Train]"):
            labels = batch["indian_label"].to(device)
            
            optimizer_phase2.zero_grad()
            logits = model(batch)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer_phase2.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Phase 2 - Epoch {epoch+1}/20 [Val]"):
                labels = batch["indian_label"].to(device)
                
                logits = model(batch)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"\nPhase 2 - Epoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Macro F1: {macro_f1:.4f}")
        
        # --- Save Best Model ---
        if macro_f1 > best_f1:
            print(f"--> Macro F1 improved from {best_f1:.4f} to {macro_f1:.4f}. Saving model...")
            best_f1 = macro_f1
            torch.save(model.state_dict(), save_path)
            
        print("-" * 50)
        
    print(f"\nTraining Complete. Best Validation Macro F1: {best_f1:.4f}")

# Usage:
model = ClipBertModel().to(device)
load_path = "../weights/clipbert/best_clipbert.pth"
model.load_state_dict(torch.load(load_path, map_location=device, weights_only=True))
train_multimodal_model(model, train_loader, val_loader, device=device)
