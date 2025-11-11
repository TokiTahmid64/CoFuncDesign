#!/usr/bin/env python3
# Fine-tune ESM2 for DNA-binding residue prediction (GLMSite-style files)

import os, re, ast, math, json, argparse, random
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

# -------------------------
# Utilities
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def log(s): print(s, flush=True)

# -------------------------
# Data: GLMSite FASTA-like parser
# Format:
# >pdb_chain
# SEQUENCEWITHOUTSPACES
# [0, 1, 0, 1, ...]  # length == len(sequence)
# (blank lines allowed between entries)
# -------------------------
def parse_glmsite_fasta(path: str) -> List[Tuple[str, List[int], str]]:
    """
    Returns list of (sequence, labels(list[int]), identifier).
    """
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            raise ValueError(f"Expected '>' header at line {i+1} in {path}")
        ident = lines[i][1:]
        if i + 2 >= len(lines):
            raise ValueError("Truncated entry near line {}".format(i+1))
        seq = lines[i+1].strip()
        # Label line might contain spaces; use ast.literal_eval on it
        label_str = lines[i+2]
        try:
            labels = ast.literal_eval(label_str)
        except Exception as e:
            raise ValueError(f"Failed to parse labels at entry {ident}: {e}")
        if not isinstance(labels, list):
            raise ValueError(f"Labels for {ident} are not a list.")
        if len(labels) != len(seq):
            raise ValueError(f"Length mismatch for {ident}: len(seq)={len(seq)} vs len(labels)={len(labels)}")
        labels = [int(x) for x in labels]
        entries.append((seq, labels, ident))
        i += 3
    return entries

# -------------------------
# Dataset & Collator
# -------------------------
class DNABindingDataset(Dataset):
    def __init__(self, entries: List[Tuple[str, List[int], str]]):
        self.data = entries

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        seq, labels, ident = self.data[idx]
        return {"seq": seq, "labels": labels, "id": ident}

def collate_fn(batch, tokenizer):
    # Tokenize as a list of strings (proteins)
    seqs = [b["seq"] for b in batch]
    enc = tokenizer(
        seqs,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
        is_split_into_words=False,
    )
    input_ids = enc["input_ids"]           # (B, Lmax)
    attention_mask = enc["attention_mask"] # (B, Lmax)
    B, Lmax = input_ids.shape

    # Build per-token labels with -100 for specials/pad
    labels_padded = torch.full((B, Lmax), fill_value=-100, dtype=torch.long)
    lengths = []
    for bi, b in enumerate(batch):
        L = len(b["labels"])
        lengths.append(L)
        # For ESM2, tokens usually: [CLS] residues [EOS] plus padding if any
        # We align residue labels to positions 1..L (exclusive of CLS/EOS)
        labels_tensor = torch.tensor(b["labels"], dtype=torch.long)
        labels_padded[bi, 1:1+L] = labels_tensor

        # Anything beyond actual residues (EOS + PAD) stays -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_padded,
        "lengths": lengths,
        "ids": [b["id"] for b in batch],
        "seqs": seqs,
    }

# -------------------------
# Model: ESM backbone + token classification head
# -------------------------
# -------------------------
# Model: ESM backbone + token classification head
# -------------------------
class ESMTokenClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, num_labels: int = 2, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        with torch.set_grad_enabled(not all(not p.requires_grad for p in self.backbone.parameters())):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state  # (B, L, H)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


# -------------------------
# Metrics
# -------------------------
def sequence_token_metrics(logits, labels, mask_valid):
    """
    logits: (B, L, 2)
    labels: (B, L) with 0/1 or -100 (ignored)
    mask_valid: (B, L) boolean for positions to score (labels != -100)
    """
    with torch.no_grad():
        preds = logits.argmax(dim=-1)  # (B, L)
        y_true = labels[mask_valid].cpu().numpy()
        y_pred = preds[mask_valid].cpu().numpy()
        if y_true.size == 0:
            return dict(acc=0.0, precision=0.0, recall=0.0, f1=0.0, mcc=0.0)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = 2 * precision * recall / max(1e-12, (precision + recall))

        # --- MCC formula ---
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-12)
        mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        return dict(acc=acc, precision=precision, recall=recall, f1=f1, mcc=mcc)


# -------------------------
# Train & Eval loops
# -------------------------
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, class_weights):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type=="cuda" else torch.float32):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, 2),
                labels.view(-1),
                ignore_index=-100,
                weight=class_weights,
            )


        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, class_weights=None):
    model.eval()
    total_loss = 0.0
    tot = dict(acc=0.0, precision=0.0, recall=0.0, f1=0.0, mcc=0.0); n_batches = 0
    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(
            logits.view(-1, 2),
            labels.view(-1),
            ignore_index=-100,
            weight=class_weights,
        )

        total_loss += float(loss.item())

        mask_valid = labels.ne(-100)
        m = sequence_token_metrics(logits, labels, mask_valid)
        for k in tot: tot[k] += m[k]
        n_batches += 1

    avg = {k: tot[k]/max(1,n_batches) for k in tot}
    avg["loss"] = total_loss / len(loader)
    return avg
    
import datetime

# -------------------------
# Logging utility
# -------------------------
def setup_logger(save_dir: str, prefix: str = "train"):
    """Create a timestamped log file and return the path."""
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(save_dir, f"{prefix}_{ts}.log")
    print(f"🧾 Logging to {log_path}")
    return open(log_path, "a", buffering=1)

# global log file handle (set later in main)
LOG_FH = None

def log(msg: str):
    """Print and save to file."""
    global LOG_FH
    print(msg, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(msg + "\n")
        LOG_FH.flush()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_fasta", type=str, required=True, help="Path to GLMSite-style training file (e.g., DNA-735-Train.fasta)")
    ap.add_argument("--test_fasta",  type=str, required=True, help="Path to GLMSite-style test file (e.g., DNA-180-test.fasta)")
    ap.add_argument("--model_name",  type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--hidden_size", type=int, default=1280, help="ESM hidden size (650M=1280, 150M=640, 35M=480)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="checkpoints_glmsite")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # --- setup logging ---
    global LOG_FH
    LOG_FH = setup_logger(args.save_dir, prefix="training")

    set_seed(args.seed)


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    # Parse data
    log("Parsing training set...")
    train_entries = parse_glmsite_fasta(args.train_fasta)
    log(f"Train entries: {len(train_entries)}")

    log("Parsing test set...")
    test_entries = parse_glmsite_fasta(args.test_fasta)
    log(f"Test entries: {len(test_entries)}")


    # --- Compute class weights from training data ---
    pos, neg = 0, 0
    for seq, labels, _ in train_entries:
        for v in labels:
            if v == 1: pos += 1
            elif v == 0: neg += 1
    total = pos + neg
    w_pos = total / (2 * pos + 1e-6)
    w_neg = total / (2 * neg + 1e-6)
    class_weights = torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device)
    log(f"Class weights -> 0: {w_neg:.3f}, 1: {w_pos:.3f}")




    # Tokenizer
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Datasets & loaders
    train_ds = DNABindingDataset(train_entries)
    test_ds  = DNABindingDataset(test_entries)

    # collator = lambda batch: collate_fn(batch, tokenizer)
    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=2, pin_memory=True)
    # test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=2, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, tokenizer),
                          num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer),
                            num_workers=0)


    # Model
    #model = ESMTokenClassifier(model_name=args.model_name, hidden_size=args.hidden_size, num_labels=2).to(device)
    
    # --- Load from local model directory (offline safe) ---
    local_model_dir = {
        "facebook/esm2_t30_150M_UR50D": "/home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/esm_model/esm2_t30_150M_UR50D_local",
        "facebook/esm2_t33_650M_UR50D": "/home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/esm_model/esm2_t33_650M_UR50D_local",
    }.get(args.model_name, args.model_name)  # fallback if user passes a path directly
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, local_files_only=True, trust_remote_code=True)
    model = ESMTokenClassifier(model_name=local_model_dir,
                               hidden_size=args.hidden_size,
                               num_labels=2).to(device)


    model_base_name = args.model_name.split("/")[1]

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_mcc = 0.0
    for epoch in range(1, args.epochs+1):
        log(f"\n=== Epoch {epoch}/{args.epochs} ===")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, class_weights)
        log(f"Train loss: {tr_loss:.4f}")

        metrics = evaluate(model, test_loader, device, class_weights)
        log(f"Eval: loss={metrics['loss']:.4f}, acc={metrics['acc']:.4f}, prec={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}, mcc={metrics['mcc']:.4f}")


        # Save best by mcc
        if metrics["mcc"] > best_mcc:
            best_mcc = metrics["mcc"]
            ckpt_path = os.path.join(args.save_dir, f"best_mcc_{model_base_name}.pt")
            torch.save({"model_state": model.state_dict(),
                        "tokenizer_name": args.model_name,
                        "hidden_size": args.hidden_size,
                        "metrics": metrics}, ckpt_path)
            log(f"✅ Saved checkpoint: {ckpt_path} with  score:  {best_mcc}"  )
            

    log("\nTraining complete.")

if __name__ == "__main__":
    main()



# python train_dna_binding.py --train_fasta /home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/dna_binding_protein/DNA-735-Train.fasta --test_fasta  /home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/dna_binding_protein/DNA-180-Test.fasta --model_name facebook/esm2_t33_650M_UR50D --hidden_size 1280 --epochs 5 --batch_size 8 --lr 1e-4