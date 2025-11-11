#!/usr/bin/env python3
# ============================================================
# Protein sequence design for DNA-binding site optimization (GLMSite format)
# ============================================================

import os, math, random, ast
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr

# ============================================================
# Global
# ============================================================
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SEARCH_MODEL_NAME = "../Finetuning/Data/esm_model/esm2_t30_150M_UR50D_local"
SEARCH_HIDDEN_SIZE = 640
SEARCH_HEAD_PATH = r"/home/md3204/Research/Toki/CoFuncDesign/Finetuning/Code/checkpoints_secondary_structure/best_acc.pt"

EVAL_MODEL_NAME = "../Finetuning/Data/esm_model/esm2_t33_650M_UR50D_local"
EVAL_HIDDEN_SIZE = 1280
EVAL_HEAD_PATH = r"/home/md3204/Research/Toki/CoFuncDesign/Finetuning/Code/checkpoints_secondary_structure/best_acc_650.pt"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# ============================================================
# FASTA Parser (GLMsite format)
# ============================================================
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

def load_secondary_structure_splits(
    csv_path: str,
    max_len: int = 512,
    sample_frac: float = 0.5,
    val_frac: float = 0.2,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess PS4 secondary structure dataset.
    Returns train_df, val_df.
    """

    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Basic cleaning
    df["input"] = df["input"].astype(str).str.upper().str.strip()
    df["dssp8"] = df["dssp8"].astype(str).str.strip()

    # Filter by sequence length
    df["seq_len"] = df["input"].apply(len)
    df = df[df["seq_len"] <= max_len].reset_index(drop=True)

    # Sort by length (reduces padding during batching)
    df = df.sort_values("seq_len").reset_index(drop=True)

    # Optionally subsample for faster debugging runs
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)

    # Split train / validation
    df_train, df_val = train_test_split(df, test_size=val_frac, random_state=seed)

    print(f"✅ Total sequences after filtering: {len(df)}")
    print(f"   ├── Train: {len(df_train)}")
    print(f"   └── Val:   {len(df_val)}")
    return df_train, df_val



# ============================================================
# Model Definition (classification)
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
    
    
def load_classifier_model(model_name, ckpt_path, hidden_size,num_labels):
    model = ESMTokenClassifier(model_name, hidden_size,num_labels).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f"✅ Loaded model from {ckpt_path}")
    return model

# ============================================================
# Optimization
# ============================================================

@torch.no_grad()
def _forward_ss_probs(model, input_ids, attn_mask, L):
    """Return per-residue class probabilities: (L, 8)."""
    logits = model(input_ids.unsqueeze(0), attn_mask.unsqueeze(0)).squeeze(0)  # (Lmax, 8)
    probs = F.softmax(logits, dim=-1)
    return probs[1:1+L]  # residues only

def compute_token_grads_hotflip_cls(model, input_ids, attn_mask, target_labels):
    """
    Multi-class HotFlip: gradient wrt input embeddings using CE loss.
    target_labels: LongTensor (L,) with class indices in [0..7]
    """
    model.zero_grad(set_to_none=True)
    emb = model.backbone.get_input_embeddings()      # (V, H)
    W = emb.weight

    saved = {}
    def hook(_, __, out):
        out.retain_grad()
        saved["emb_out"] = out
    h = emb.register_forward_hook(hook)

    logits_full = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0)).squeeze(0)  # (Lmax, 8)
    L = target_labels.size(0)
    logits = logits_full[1:1+L]  # residues only

    loss = F.cross_entropy(logits, target_labels)    # <-- CE for multi-class
    loss.backward()

    emb_grad = saved["emb_out"].grad.detach().squeeze(0)  # (Lmax, H)
    h.remove()

    scores = - emb_grad @ W.T   # (Lmax, V)
    return scores, float(loss.item()), logits.detach()

@torch.no_grad()
def evaluate_seq_cls(model, input_ids, attn_mask, target_labels):
    """Return CE loss on residues only."""
    logits_full = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0)).squeeze(0)
    L = target_labels.size(0)
    logits = logits_full[1:1+L]
    loss = F.cross_entropy(logits, target_labels)
    return logits.softmax(-1), float(loss.item())


def gradient_greedy_search(model_search, model_eval, tokenizer, input_ids, attn_mask,
                           target_vec, allowed_tokens, iters=400, k=4, B=32, patience=50):
    fixed_indices = {0, input_ids.size(0) - 1}
    best_seq = input_ids.clone()
    _, best_loss = evaluate_seq_cls(model_eval, best_seq, attn_mask, target_vec)
    last_best = 0

    allowed_mask = torch.full((best_seq.size(0), model_search.backbone.get_input_embeddings().weight.size(0)),
                              True, device=DEVICE, dtype=torch.bool)
    allowed_mask[:, allowed_tokens] = False

    for t in tqdm(range(iters), desc="Optimizing"):
        scores, _, _ = compute_token_grads_hotflip_cls(model_search, best_seq, attn_mask, target_vec)
        scores = scores.masked_fill(allowed_mask, float("-inf"))
        tk = scores.topk(k, dim=1)
        tk_idx, tk_val = tk.indices, tk.values
        p_replace = max(0.25, 0.5 * math.exp(-t / 50))

        cands, losses = [], []
        for _ in range(B):
            cand = best_seq.clone()
            rand_mask = (torch.rand_like(cand.float()) < p_replace)
            for i in range(cand.size(0)):
                if i in fixed_indices or not rand_mask[i]: continue
                probs = F.softmax(tk_val[i], dim=0)
                j = torch.multinomial(probs, 1)
                cand[i] = tk_idx[i, j]
            cands.append(cand)

        for c in cands:
            _, l = evaluate_seq_cls(model_eval, c, attn_mask, target_vec)
            losses.append(l)

        idx = int(np.argmin(losses))
        if losses[idx] < best_loss:
            best_seq, best_loss, last_best = cands[idx], losses[idx], t
        elif t - last_best >= patience:
            print(f"⏹ Early stop at {t}")
            break

    final_probs, _ = evaluate_seq_cls(model_eval, best_seq, attn_mask, target_vec)
    decoded = tokenizer.decode(best_seq, skip_special_tokens=True).replace(" ", "")
    return decoded, final_probs



from sklearn.metrics import matthews_corrcoef


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("🔹 Loading models...")
    search_model = load_classifier_model(SEARCH_MODEL_NAME, SEARCH_HEAD_PATH, SEARCH_HIDDEN_SIZE,num_labels=8)
    eval_model = load_classifier_model(EVAL_MODEL_NAME, EVAL_HEAD_PATH, EVAL_HIDDEN_SIZE,num_labels=8)
    tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL_NAME, local_files_only=True)
    allowed_tokens = tokenizer.convert_tokens_to_ids(AA_LIST)

    csv_path = "/home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/secondary_structure/data.csv"
    train_df, val_df = load_secondary_structure_splits(csv_path)
    val_df = val_df[["chain_id","input", "dssp8", "seq_len"]].rename(columns={"ident":"chain_id","input": "seq", "dssp8": "labels"})
    # sort the validation entries by 'seq_len' column
    sorted_entries = val_df.sort_values("seq_len").reset_index(drop=True).to_dict(orient="records")
    # drop 'seq_len' from entries
    for entry in sorted_entries:
        entry.pop("seq_len", None)

        # drop 'seq_len' from entries

    # take only the first 50 shortest entries for demo
    entries = sorted_entries[:50]
    all_ids = [entry["chain_id"] for entry in entries]
    all_seqs = [entry["seq"] for entry in entries]
    all_labels = [entry["labels"] for entry in entries]
    DSSP8_MAP = {c: i for i, c in enumerate("HBEGITS-")}  # '-' or 'C' for coil

    results = []
    for idx, (ident, seq, labels) in enumerate(zip(all_ids, all_seqs, all_labels)):  # demo subset

        labels = [DSSP8_MAP.get(c, DSSP8_MAP["-"]) for c in labels ]
        target_vec = torch.tensor(labels, dtype=torch.long, device=DEVICE)
        L = len(seq)
        print(f"\n🧬 Designing {ident} ({idx+1}/{len(entries)}) | L={L}")

        enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids, attn_mask = enc["input_ids"][0].to(DEVICE), enc["attention_mask"][0].to(DEVICE)

        # -----------------------------
        # Ground-truth metrics
        # -----------------------------
        with torch.no_grad():
            probs_gt = _forward_ss_probs(eval_model, input_ids, attn_mask, L)
            # print(probs_gt)
        # pred_gt = (probs_gt > 0.5).float().cpu().numpy()
        preds = probs_gt.argmax(-1)

        y_true = target_vec.cpu().numpy()
        # convert preds to numpy
        preds = preds.cpu().numpy()


        acc_gt = ((preds == y_true)).mean()
        mcc_gt = matthews_corrcoef(y_true, preds)
        print(f"📊 Ground seq: ACC={acc_gt:.3f}, MCC={mcc_gt:.3f}")

        # -----------------------------
        # Initialize random sequence
        # -----------------------------
        rand_core = torch.tensor(np.random.choice(allowed_tokens, L), device=DEVICE)
        dummy = "A" * L
        enc_d = tokenizer(dummy, return_tensors="pt", add_special_tokens=True)
        input_rand = enc_d["input_ids"][0].to(DEVICE)
        attn_rand = enc_d["attention_mask"][0].to(DEVICE)
        input_rand[1:1+L] = rand_core[:L]
        # break
        # -----------------------------
        # Optimize sequence
        # -----------------------------
        designed_seq, probs_new = gradient_greedy_search(
            model_search=search_model,
            model_eval=eval_model,
            tokenizer=tokenizer,
            input_ids=input_rand,
            attn_mask=attn_rand,
            target_vec=target_vec,
            allowed_tokens=allowed_tokens,
            iters=500,
        )

        pred_new = probs_new.argmax(-1).cpu().numpy()
        acc_new = (pred_new == y_true).mean()
        mcc_new = matthews_corrcoef(y_true, pred_new)

        print(f"✅ Designed seq: {designed_seq[:60]}...")
        print(f"🔹 ACC={acc_new:.3f}, ΔACC={acc_new-acc_gt:+.3f}, "
              f"MCC={mcc_new:.3f}, ΔMCC={mcc_new-mcc_gt:+.3f}")

        results.append({
            "id": ident,
            "original_seq": seq,
            "designed_seq": designed_seq,
            "length": L,
            "acc_original": acc_gt,
            "acc_generated": acc_new,
            "delta_acc": acc_new - acc_gt,
            "mcc_original": mcc_gt,
            "mcc_generated": mcc_new,
            "delta_mcc": mcc_new - mcc_gt,
        })

    df = pd.DataFrame(results)
    df.to_csv("designed_sequences_binding_results.csv", index=False)
    print("\n💾 Saved results to designed_sequences_binding_results.csv")
    print(f"📈 Mean ΔACC: {df['delta_acc'].mean():+.3f}, Mean ΔMCC: {df['delta_mcc'].mean():+.3f}")
