# ============================================================
# Protein sequence design for target solubility using dual ESM2 models
# ============================================================

import os, math, random, re
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ============================================================
# Global 
# ============================================================
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

#BASE_SEARCH_MODEL_NAME = r"D:\Toki\Academic\First_Year\COS551\COS551\Project\CoFuncDesign\Codes\Finetuning\esm_model\esm2_t30_150M_UR50D_local"
#BASE_EVAL_MODEL_NAME = r"D:\Toki\Academic\First_Year\COS551\COS551\Project\CoFuncDesign\Codes\Finetuning\esm_model\esm2_t33_650M_UR50D_local"



SEARCH_MODEL_NAME = "/home/md3204/Research/Toki/CoFuncDesign/Finetuning/Data/esm_model/esm2_t30_150M_UR50D_local"
SEARCH_HIDDEN_SIZE = 640
SEARCH_HEAD_PATH = "/home/md3204/Research/Toki/CoFuncDesign/Finetuning/Code/checkpoints_solubility/best_pcc_esm2_t30_150M_UR50D.pt"

EVAL_MODEL_NAME = "/home/md3204/Research/Toki/CoFuncDesign/Finetuning/Data/esm_model/esm2_t33_650M_UR50D_local"
EVAL_HIDDEN_SIZE = 1280
EVAL_HEAD_PATH = "/home/md3204/Research/Toki/CoFuncDesign/Finetuning/Code/checkpoints_solubility/best_pcc_esm2_t33_650M_UR50D.pt"
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
VOCAB_SIZE = len(AA_LIST)


# ============================================================
# Global 
# ============================================================

print("Starting...")


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def aa_token_ids(tokenizer) -> List[int]:
    """Token IDs for canonical amino acids."""
    return tokenizer.convert_tokens_to_ids(AA_LIST)



class ESMRegressionHead(nn.Module):
    def __init__(self, model_name, hidden_size):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
            # inputs_embeds=inputs_embeds
        )

        # print("Output last_hidden_state shape:", out.last_hidden_state.shape)
        x = self.dropout(out.last_hidden_state)
        y_hat = self.regressor(x).squeeze(-1)
        return y_hat


# ============================================================
# Model loading
# ============================================================
def load_regression_model(model_name, ckpt_path, hidden_size, device=DEVICE):
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_model_dir = model_name
    model = ESMRegressionHead(local_model_dir, hidden_size=hidden_size).to(device)
    print("Base model loaded")
    checkpoint = torch.load(
        ckpt_path,
    map_location=device
    )

    # if it's a dict with 'model_state' key, load that
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    else:
        model.load_state_dict(checkpoint,strict=False)

    print("✅ Loaded trained model weights.")
    model.eval()
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    return model

def load_dual_models(device=DEVICE):
    print("🔹 Loading search model...")
    search_model = load_regression_model(SEARCH_MODEL_NAME, SEARCH_HEAD_PATH, SEARCH_HIDDEN_SIZE, device)
    print("🔹 Loading evaluation model...")
    eval_model= load_regression_model(EVAL_MODEL_NAME, EVAL_HEAD_PATH, EVAL_HIDDEN_SIZE, device)
    return search_model, eval_model


# ============================================================
# Gradient Optimization
# ============================================================
@torch.no_grad()
def _forward_preds_vector(model, input_ids, attn_mask, target_vec_len):
    """
    Forward pass that returns per-residue predictions aligned to residues only:
    positions 1..L (skip [CLS] at 0 and [EOS]/pad beyond L).
    """
    # (1, Lmax) -> (1, Lmax)
    preds_full = model(input_ids.unsqueeze(0), attn_mask.unsqueeze(0)).squeeze(0)  # (Lmax,)
    # Use exactly the length of target vector (L residues)
    preds_vec = preds_full[1:1 + target_vec_len]  # (L,)
    return preds_vec

def compute_token_grads_hotflip(model, input_ids, attn_mask, target_vec):
    """
    HotFlip-style gradient: get grad wrt *input embeddings* per position,
    then score all tokens by -grad @ E^T. Returns (scores, loss, preds_vec).
      - scores: (Lmax, V) float tensor; more positive = better (lower loss)
    """
    model.zero_grad(set_to_none=True)
    emb_layer = model.backbone.get_input_embeddings()   # nn.Embedding(V, H)
    W = emb_layer.weight                                # (V, H)

    # Hook to capture the *output* of the embedding layer and keep grad
    saved = {}
    def save_emb_out(module, inp, out):
        # out: (B, Lmax, H)
        out.retain_grad()
        saved["emb_out"] = out

    h = emb_layer.register_forward_hook(save_emb_out)

    # Forward (standard input_ids path)
    preds_full = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0)).squeeze(0)  # (Lmax,)
    L = target_vec.size(0)
    preds_vec = preds_full[1:1+L]  # residues only

    loss = F.mse_loss(preds_vec, target_vec)
    loss.backward()

    # Gradient wrt input embeddings at each position
    emb_grad = saved["emb_out"].grad.detach().squeeze(0)  # (Lmax, H)
    h.remove()

    # scores[i, t] = - grad_i dot E_t   (drop the E_current term since it's a constant shift)
    scores = - emb_grad @ W.T  # (Lmax, V)

    return scores, float(loss.item()), preds_vec.detach()
@torch.no_grad()
def evaluate_seq(model, input_ids, attn_mask, target_vec):
    preds_full = model(input_ids=input_ids.unsqueeze(0), attention_mask=attn_mask.unsqueeze(0)).squeeze(0)
    preds_vec = preds_full[1:1+target_vec.size(0)]
    loss = F.mse_loss(preds_vec, target_vec)
    return preds_vec.detach(), float(loss.item())

def gradient_greedy_search(
    model_search, model_eval, tokenizer,
    input_ids, attn_mask, target_vec,   # target_vec: (L,)
    allowed_tokens,
    iters=200, k=4, B=32, start_thresh=0.5, end_thresh=0.25, patience=80,
):
    fixed_indices = {0, input_ids.size(0) - 1}  # keep CLS/EOS fixed
    best_seq = input_ids.clone().to(torch.long)

    _, best_loss = evaluate_seq(model_eval, best_seq, attn_mask, target_vec)
    print(f"Initial eval MSE: {best_loss:.6f} | target_len={target_vec.size(0)}")
    last_best = 0

    # mask illegal tokens once
    allowed_mask = torch.full((best_seq.size(0), model_search.backbone.get_input_embeddings().weight.size(0)),
                              True, device=best_seq.device, dtype=torch.bool)
    allowed_mask[:, allowed_tokens] = False  # False == allowed

    for t in tqdm(range(iters), desc="Gradient-Greedy"):
        # HotFlip scores from the *search* model
        scores, _, _ = compute_token_grads_hotflip(model_search, best_seq, attn_mask, target_vec)  # (Lmax, V)

        # block disallowed tokens
        scores = scores.masked_fill(allowed_mask, float("-inf"))

        # take top-k per position
        topk = scores.topk(k, dim=1)
        tk_idx, tk_val = topk.indices, topk.values

        decay = 50  # decay rate for exponential decay

        # annealed replacement prob
        # p_replace = max(end_thresh, start_thresh - (start_thresh - end_thresh) * (t / max(1, iters)))
        p_replace = max(end_thresh, start_thresh * math.exp(-t / decay))


        # generate B candidates
        cands = []
        for _ in range(B):
            cand = best_seq.clone()
            rand_mask = (torch.rand_like(cand.float()) < p_replace)
            for i in range(cand.size(0)):
                if i in fixed_indices or not rand_mask[i]:
                    continue
                # sample among top-k proposals at position i
                probs = F.softmax(tk_val[i], dim=0)
                j = torch.multinomial(probs, 1)
                cand[i] = tk_idx[i, j]
            cands.append(cand)

        # evaluate with the *eval* model
        losses = []
        for c in cands:
            _, l = evaluate_seq(model_eval, c, attn_mask, target_vec)
            losses.append(l)

        idx = int(np.argmin(losses))
        if losses[idx] < best_loss:
            best_seq, best_loss, last_best = cands[idx], losses[idx], t
        elif t - last_best >= patience:
            print(f"Early stop at {t}")
            break

    final_preds, final_loss = evaluate_seq(model_eval, best_seq, attn_mask, target_vec)
    decoded = tokenizer.decode(best_seq, skip_special_tokens=True).replace(" ", "")
    return decoded, final_preds, final_loss



# ============================================================
# Dataset reading (ProteinGLUE solubility format)
# ============================================================
import pandas as pd
import re
import numpy as np

def parse_multiline_array(s: str):
    """
    Convert '[1.38 0.8199 1.13 ...]' into a list of floats.
    Handles multiline strings and inconsistent spaces.
    """
    if pd.isna(s) or not isinstance(s, str):
        return []
    s = s.replace('\n', ' ').replace('[', '').replace(']', '')
    parts = re.split(r'\s+', s.strip())
    vals = []
    for x in parts:
        if x == '':
            continue
        try:
            vals.append(float(x))
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

# ============================================================
# Main
# ============================================================
# set_seed(42)
search_model, eval_model = load_dual_models(DEVICE)


import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
import numpy as np

print("Loading data...")
# --- load dataset ---
df = load_proteinglue_csv("../Finetuning/Data/solubility_prediction/asabu_test.csv")

print("Loaded data...")
# sort by sequence length for stability
df["seq_len"] = df["sequence"].apply(len)

# only keep sequences with length below 200 for demo
df = df[df["seq_len"] <= 200].reset_index(drop=True)
# shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.head(100)  # only first few for demo
tokenizer = AutoTokenizer.from_pretrained(SEARCH_MODEL_NAME, local_files_only=True)

allowed_tokens = aa_token_ids(tokenizer)


import pandas as pd

results = []  # <-- store per-sequence results

for idx, row in df.iterrows():
    target_vec = torch.tensor(row["solvent_accessibility"], dtype=torch.float32, device=DEVICE)
    seq_len = len(target_vec)

    print(f"\n🧩 Designing sequence {idx+1}/{len(df)} | Length: {seq_len}")

    # -------------------------------
    # 1️⃣ Evaluate ground-truth sequence
    # -------------------------------
    seq_native = row["sequence"]
    enc_gt = tokenizer(seq_native, return_tensors="pt", add_special_tokens=True)
    gt_input = enc_gt["input_ids"][0].to(DEVICE)
    gt_mask  = enc_gt["attention_mask"][0].to(DEVICE)

    with torch.no_grad():
        pred_gt = _forward_preds_vector(eval_model, gt_input, gt_mask, target_vec.size(0))
        loss_gt = F.mse_loss(pred_gt, target_vec)

    pred_gt_np = pred_gt.cpu().numpy()
    tgt_np     = target_vec.cpu().numpy()
    pcc_gt     = pearsonr(pred_gt_np, tgt_np)[0]
    mae_gt     = np.mean(np.abs(pred_gt_np - tgt_np))
    mse_gt     = np.mean((pred_gt_np - tgt_np)**2)

    print(f"📊 Ground sequence:")
    print(f"   PCC={pcc_gt:.4f} | MAE={mae_gt:.4f} | MSE={mse_gt:.6f}")

    # -------------------------------
    # 2️⃣ Initialize random sequence
    # -------------------------------
    rand_core = torch.tensor(np.random.choice(allowed_tokens, seq_len), device=DEVICE)
    dummy = "A" * seq_len
    enc = tokenizer(dummy, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"][0].to(DEVICE)
    attn_mask = enc["attention_mask"][0].to(DEVICE)

    input_ids = input_ids.to(torch.long)
    attn_mask = attn_mask.to(torch.long)
    input_ids[1:1+seq_len] = rand_core[:seq_len]

    # -------------------------------
    # 3️⃣ Run gradient-based sequence design
    # -------------------------------
    designed_seq, pred_vec, loss = gradient_greedy_search(
        model_search=search_model,
        model_eval=eval_model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attn_mask=attn_mask,
        target_vec=target_vec,
        allowed_tokens=allowed_tokens,
        iters=1000,
    )

    # -------------------------------
    # 4️⃣ Evaluate generated sequence
    # -------------------------------
    pred_np = pred_vec.cpu().numpy()
    pcc = pearsonr(pred_np, tgt_np)[0]

    pcc_gen = pearsonr(pred_gt_np, pred_np)[0]  # generator model performance
    mae = np.mean(np.abs(pred_np - tgt_np))
    mse = np.mean((pred_np - tgt_np)**2)

    print(f"✅ Designed sequence: {designed_seq}")
    print(f"🔹 PCC={pcc:.4f} | MAE={mae:.4f} | MSE={mse:.6f}")
    print(f"📈 ΔPCC = {pcc - pcc_gt:+.4f}, ΔMAE = {mae - mae_gt:+.4f}")

    # -------------------------------
    # 5️⃣ Save result for this sequence
    # -------------------------------
    results.append({
        "index": idx,
        "original_sequence": seq_native,
        "designed_sequence": designed_seq,
        "length": seq_len,
        "eval_model_performance": pcc_gt, # real output vs real seq prediction from eval model
        "generator_model_performance": pcc_gen, # generated seq prediction from eval model vs real seq prediction from eval model
        "robustness": pcc, # real output vs generated seq prediction from eval model
            
    })

# -----------------------------------------------------
# ✅ After the loop: convert to DataFrame and save
# -----------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("designed_sequences_solubility_results.csv", index=False)
print("\n💾 Saved all designed sequences to designed_sequences_solubility_results.csv")

