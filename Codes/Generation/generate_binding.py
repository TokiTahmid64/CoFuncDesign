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
SEARCH_HEAD_PATH = "../Finetuning/Code/checkpoints_glmsite/best_mcc_esm2_t30_150M_UR50D.pt"

EVAL_MODEL_NAME = "../Finetuning/Data/esm_model/esm2_t33_650M_UR50D_local"
EVAL_HIDDEN_SIZE = 1280
EVAL_HEAD_PATH = "../Finetuning/Code/checkpoints_glmsite/best_mcc_esm2_t33_650M_UR50D.pt"

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

# ============================================================
# FASTA Parser (GLMsite format)
# ============================================================
def parse_glmsite_fasta(path: str) -> List[Tuple[str, List[int], str]]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            raise ValueError(f"Expected '>' header at line {i+1} in {path}")
        ident = lines[i][1:]
        seq = lines[i+1].strip()
        labels = ast.literal_eval(lines[i+2])
        if len(labels) != len(seq):
            raise ValueError(f"Length mismatch for {ident}")
        entries.append((seq, [int(x) for x in labels], ident))
        i += 3
    print(f"✅ Loaded {len(entries)} FASTA entries from {path}")
    return entries

# ============================================================
# Model Definition (classification)
# ============================================================
class ESMTokenClassifier(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels=2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(out.last_hidden_state)
        logits = self.classifier(x)  # (B, L, 2)
        return logits

def load_classifier_model(model_name, ckpt_path, hidden_size):
    model = ESMTokenClassifier(model_name, hidden_size).to(DEVICE)
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
def _forward_binding_probs(model, input_ids, attn_mask, L):
    logits = model(input_ids.unsqueeze(0), attn_mask.unsqueeze(0)).squeeze(0)  # (Lmax, 2)
    probs = F.softmax(logits, dim=-1)[:, 1]  # binding probability
    return probs[1:1+L]  # residues only

def compute_token_grads_hotflip(model, input_ids, attn_mask, target_vec):
    model.zero_grad(set_to_none=True)
    emb_layer = model.backbone.get_input_embeddings()
    W = emb_layer.weight
    saved = {}

    def save_emb_out(module, inp, out):
        out.retain_grad()
        saved["emb_out"] = out
    h = emb_layer.register_forward_hook(save_emb_out)

    logits = model(input_ids.unsqueeze(0), attn_mask.unsqueeze(0)).squeeze(0)
    L = target_vec.size(0)
    probs = F.softmax(logits[1:1+L], dim=-1)[:, 1]
    loss = F.binary_cross_entropy(probs, target_vec)
    loss.backward()

    emb_grad = saved["emb_out"].grad.detach().squeeze(0)
    h.remove()
    scores = -emb_grad @ W.T
    return scores, float(loss.item()), probs.detach()

@torch.no_grad()
def evaluate_seq(model, input_ids, attn_mask, target_vec):
    logits = model(input_ids.unsqueeze(0), attn_mask.unsqueeze(0)).squeeze(0)
    probs = F.softmax(logits, dim=-1)[:, 1]
    probs = probs[1:1+target_vec.size(0)]
    loss = F.binary_cross_entropy(probs, target_vec)
    return probs.detach(), float(loss.item())

def gradient_greedy_search(model_search, model_eval, tokenizer, input_ids, attn_mask,
                           target_vec, allowed_tokens, iters=400, k=4, B=32, patience=50):
    fixed_indices = {0, input_ids.size(0) - 1}
    best_seq = input_ids.clone()
    _, best_loss = evaluate_seq(model_eval, best_seq, attn_mask, target_vec)
    last_best = 0

    allowed_mask = torch.full((best_seq.size(0), model_search.backbone.get_input_embeddings().weight.size(0)),
                              True, device=DEVICE, dtype=torch.bool)
    allowed_mask[:, allowed_tokens] = False

    for t in tqdm(range(iters), desc="Optimizing"):
        scores, _, _ = compute_token_grads_hotflip(model_search, best_seq, attn_mask, target_vec)
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
            _, l = evaluate_seq(model_eval, c, attn_mask, target_vec)
            losses.append(l)

        idx = int(np.argmin(losses))
        if losses[idx] < best_loss:
            best_seq, best_loss, last_best = cands[idx], losses[idx], t
        elif t - last_best >= patience:
            print(f"⏹ Early stop at {t}")
            break

    final_probs, _ = evaluate_seq(model_eval, best_seq, attn_mask, target_vec)
    decoded = tokenizer.decode(best_seq, skip_special_tokens=True).replace(" ", "")
    return decoded, final_probs



from sklearn.metrics import matthews_corrcoef

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("🔹 Loading models...")
    search_model = load_classifier_model(SEARCH_MODEL_NAME, SEARCH_HEAD_PATH, SEARCH_HIDDEN_SIZE)
    eval_model = load_classifier_model(EVAL_MODEL_NAME, EVAL_HEAD_PATH, EVAL_HIDDEN_SIZE)
    tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL_NAME, local_files_only=True)
    allowed_tokens = tokenizer.convert_tokens_to_ids(AA_LIST)

    fasta_path = "/home/mt3204/Toki/Academic_Projects/COS551/CoFuncDesign/Finetuning/Data/dna_binding_protein/DNA-180-Test.fasta"
    entries = parse_glmsite_fasta(fasta_path)

    sorted_entries = sorted(entries, key=lambda x: len(x[0]))

    # take only the first 50 shortest entries for demo
    entries = sorted_entries[:50]
    
    
    results = []
    for idx, (seq, labels, ident) in enumerate(entries[:30]):  # demo subset
        target_vec = torch.tensor(labels, dtype=torch.float32, device=DEVICE)
        L = len(seq)
        print(f"\n🧬 Designing {ident} ({idx+1}/{len(entries)}) | L={L}")

        enc = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
        input_ids, attn_mask = enc["input_ids"][0].to(DEVICE), enc["attention_mask"][0].to(DEVICE)

        # -----------------------------
        # Ground-truth metrics
        # -----------------------------
        with torch.no_grad():
            probs_gt = _forward_binding_probs(eval_model, input_ids, attn_mask, L)
        pred_gt = (probs_gt > 0.5).float().cpu().numpy()
        y_true = target_vec.cpu().numpy()

        acc_gt = (pred_gt == y_true).mean()
        mcc_gt = matthews_corrcoef(y_true, pred_gt)
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

        pred_new = (probs_new > 0.5).float().cpu().numpy()
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
