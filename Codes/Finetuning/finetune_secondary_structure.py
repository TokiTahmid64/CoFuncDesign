#!/usr/bin/env python3
# Train ESM2 for secondary structure prediction (PS4 dataset)

import os, math, random
import pandas as pd
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ============================================================
# Globals
# ============================================================
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {DEVICE}")

MODEL_NAME = "/home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/esm_model/esm2_t33_650M_UR50D_local"
HIDDEN_SIZE = 1280
NUM_LABELS = 8  # DSSP8 classes (H, B, E, G, I, T, S, C)
SAVE_DIR = "checkpoints_secondary_structure"
os.makedirs(SAVE_DIR, exist_ok=True)

# DSSP8 mapping to numeric
DSSP8_MAP = {c: i for i, c in enumerate("HBEGITS-")}  # '-' or 'C' for coil


# ============================================================
# Dataset
# ============================================================
class SecondaryStructureDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["input"].strip()
        ss = row["dssp8"].strip()

        # convert labels to indices
        labels = [DSSP8_MAP.get(c, DSSP8_MAP["-"]) for c in ss]
        return {"seq": seq, "labels": labels}


def collate_fn(batch, tokenizer):
    seqs = [b["seq"] for b in batch]
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    B, Lmax = input_ids.shape

    labels_padded = torch.full((B, Lmax), fill_value=-100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["labels"])
        labels_padded[i, 1:1+L] = torch.tensor(b["labels"][:L])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_padded,
    }


# ============================================================
# Model
# ============================================================
class ESMTokenClassifier(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)
        logits = self.classifier(x)
        return logits


# ============================================================
# Metrics
# ============================================================
def calc_metrics(logits, labels, mask):
    preds = logits.argmax(-1)
    y_true = labels[mask].cpu().numpy()
    y_pred = preds[mask].cpu().numpy()

    acc = (y_true == y_pred).mean()
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-12)
    mcc = ((tp*tn)-(fp*fn))/denom if denom > 0 else 0.0
    return acc, mcc


# ============================================================
# Train & Eval
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attn_mask)
            loss = F.cross_entropy(
                logits.view(-1, NUM_LABELS),
                labels.view(-1),
                ignore_index=-100
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, total_acc, total_mcc = 0, 0, 0
    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attn_mask)
        loss = F.cross_entropy(
            logits.view(-1, NUM_LABELS),
            labels.view(-1),
            ignore_index=-100
        )
        total_loss += loss.item()

        mask = labels != -100
        acc, mcc = calc_metrics(logits, labels, mask)
        total_acc += acc
        total_mcc += mcc

    n = len(loader)
    return total_loss/n, total_acc/n, total_mcc/n


# ============================================================
# Main
# ============================================================

df = pd.read_csv(r"/home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/secondary_structure/data.csv")  # <-- your dataset

# sort the dataset based on sequence length to minimize padding
df["seq_len"] = df["input"].apply(len)
df = df.sort_values("seq_len").reset_index(drop=True)

# take only sequences of length <= 512 for ESM2
df = df[df["seq_len"] <= 512]
# make the input column all uppercase
df["input"] = df["input"].str.upper()

# randomly take 50% of the data for faster training
#df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)


df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
train_ds = SecondaryStructureDataset(df_train)
val_ds = SecondaryStructureDataset(df_val)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, tokenizer))
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, tokenizer))

model = ESMTokenClassifier(MODEL_NAME, HIDDEN_SIZE, NUM_LABELS).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader)*3)
scaler = torch.cuda.amp.GradScaler()



best_acc = -1
for epoch in range(10):
    print(f"\nEpoch {epoch+1}")
    tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, DEVICE)
    val_loss, val_acc, val_mcc = evaluate(model, val_loader, DEVICE)
    print(f"Train Loss={tr_loss:.4f} | Val Loss={val_loss:.4f} | ACC={val_acc:.4f} | MCC={val_mcc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_acc_650.pt")
        print(f"✅ Saved best model (ACC={best_acc:.4f})")



