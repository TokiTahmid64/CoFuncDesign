import os
import re
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ------------------------------------------------------
# Utility: Parse multi-line numeric arrays
# ------------------------------------------------------
def parse_multiline_array(s: str):
    """
    Convert multi-line strings like:
    '[1.38 0.81999999 1.13 ...]' or '[0 0 1 1 0]'
    into Python lists of floats or ints.
    Handles inconsistent spacing/newlines.
    """
    if pd.isna(s) or not isinstance(s, str):
        return []

    # Remove brackets and newlines
    s = s.replace('\n', ' ').replace('[', '').replace(']', '')
    # Split on whitespace
    parts = re.split(r'\s+', s.strip())
    # Try to parse as float, fallback to int
    vals = []
    for x in parts:
        if x == '':
            continue
        try:
            vals.append(float(x))
        except ValueError:
            try:
                vals.append(int(x))
            except ValueError:
                pass
    return vals


def load_proteinglue_csv(path):
    """Load and parse ProteinGLUE solvent accessibility CSV."""
    print(f"\n📂 Loading: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse numeric arrays
    df["solvent_accessibility"] = df["solvent_accessibility"].apply(parse_multiline_array)
    df["buried"] = df["buried"].apply(parse_multiline_array)
    df["sequence"] = (
        df["sequence"].astype(str)
        .str.replace("'", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("b", "", regex=False)
        .str.strip()
    )

    # print(df.head(2))


    # Remove rows where sequence and solvent_accessibility lengths differ
    mismatched = df[df.apply(
        lambda r: abs(len(r["sequence"]) - len(r["solvent_accessibility"])) > 2,
        axis=1
    )]
    df["sequence"] = df["sequence"].apply(lambda s: s[1:-1] if len(s) > 2 else s)


    if len(mismatched) > 0:
        print(f"⚠️ {len(mismatched)} rows removed (sequence length mismatch).")

    df = df.drop(mismatched.index).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} valid proteins.")
    return df
# ------------------------------------------------------
# Dataset
# ------------------------------------------------------
class SolventAccessibilityDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["sequence"]
        y = torch.tensor(row["solvent_accessibility"], dtype=torch.float)
        return {"seq": seq, "labels": y}

# ------------------------------------------------------
# Collate function
# ------------------------------------------------------
def collate_fn(batch, tokenizer):
    seqs = [b["seq"] for b in batch]
    labels = [b["labels"] for b in batch]
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    max_len = input_ids.size(1)
    labels_padded = torch.full((len(batch), max_len), -100.0)
    for i, y in enumerate(labels):
        labels_padded[i, 1:1+len(y)] = y[:max_len-2]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_padded
    }

# ------------------------------------------------------
# Model
# ------------------------------------------------------
class ESMRegressionHead(nn.Module):
    def __init__(self, model_name, hidden_size):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)
        y_hat = self.regressor(x).squeeze(-1)
        return y_hat

# ------------------------------------------------------
# Train & evaluate
# ------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad(set_to_none=True)
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            preds = model(**inputs)
            mask = labels != -100
            loss = F.mse_loss(preds[mask], labels[mask])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Eval"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        preds = model(**inputs)
        mask = labels != -100
        loss = F.mse_loss(preds[mask], labels[mask])
        total_loss += loss.item()
    return total_loss / len(loader)

# ------------------------------------------------------
# Main training loop
# ------------------------------------------------------
def main():
    local_models = {
        "facebook/esm2_t30_150M_UR50D": "D:\\Toki\\Academic\\First_Year\\COS551\\COS551\\Project\\CoFuncDesign\\Codes\\Finetuning\\esm_model\\esm2_t30_150M_UR50D_local",
        "facebook/esm2_t33_650M_UR50D": "D:\\Toki\\Academic\\First_Year\\COS551\\COS551\\Project\\CoFuncDesign\\Codes\\Finetuning\\esm_model\\esm2_t33_650M_UR50D_local"
    }

    model_name = "facebook/esm2_t33_650M_UR50D"
    model_dir = local_models[model_name]
    hidden_size = 1280 if "650M" in model_name else 640

    train_df = load_proteinglue_csv("../../Datasets/Finetuning/solvent_accessibility/asabu_training.csv")
    val_df   = load_proteinglue_csv("../../Datasets/Finetuning/solvent_accessibility/asabu_validation.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train_ds = SolventAccessibilityDataset(train_df)
    val_ds   = SolventAccessibilityDataset(val_df)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMRegressionHead(model_dir, hidden_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    num_training_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    best_val = 1e9
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), f"best_esm_sap.pt")
            print(f"✅ Saved best model (Val Loss = {best_val:.4f})")

if __name__ == "__main__":
    main()
