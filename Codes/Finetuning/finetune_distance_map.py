#==========================================================
# Code to finetune Protein distance map from sequence using ESM-150m and ESM-650m models.

# Written by: Lana Glisic

#==========================================================

import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class NPZSequenceDistanceDataset(Dataset):
    def __init__(self, npz_dir: str):
        self.files = sorted(
            f for f in os.listdir(npz_dir) if f.endswith(".npz")
        )
        self.npz_dir = npz_dir

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.npz_dir, self.files[idx])
        data = np.load(path, allow_pickle=True)
        sequence = str(data["sequence"])
        distance_map = data["distance_map"].astype(np.float32)
        return sequence, distance_map


class SequenceDistanceDataset(Dataset):
    def __init__(self, sequences, distance_maps):
        self.sequences = sequences
        self.maps = distance_maps

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.maps[idx]


def collate_fn(batch: list[Tuple[str, np.ndarray]], tokenizer):
    sequences = [item[0] for item in batch]
    distance_maps_list = [item[1] for item in batch]
    lengths = [len(seq) for seq in sequences]

    enc = tokenizer(sequences, return_tensors="pt", padding=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    B, Lmax = input_ids.shape
    padded_distance_maps = torch.zeros((B, Lmax, Lmax), dtype=torch.float32)

    for i, dist in enumerate(distance_maps_list):
        L = dist.shape[0]
        dist = torch.as_tensor(dist, dtype=torch.float32)
        padded_distance_maps[i, 1:1+L, 1:1+L] = dist

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "distance_maps": padded_distance_maps,
        "lengths": lengths
    }


class DistanceMapPredictor(nn.Module):
    def __init__(self, esm_model, hidden_size: int = 1280, proj_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.esm = esm_model
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, proj_size),
            nn.ReLU(),
            nn.Dropout(dropout) # Significant improvement
        )

        # 2 * proj_size -> 512 -> 256 -> 64 -> 1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2*proj_size, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, lengths: list):
        # ESM embeddings
        h = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.proj(h)

        B, L, C = h.shape

        hi = h.unsqueeze(2).expand(-1, -1, L, -1)
        hj = h.unsqueeze(1).expand(-1, L, -1, -1)
        pairwise = torch.cat([hi, hj], dim=-1)
        pairwise = pairwise.permute(0, 3, 1, 2)

        dist_pred = self.conv_layers(pairwise).squeeze(1)

        mask = torch.zeros_like(dist_pred)
        for i, l in enumerate(lengths):
            mask[i, 1:l+1, 1:l+1] = 1.0

        dist_pred = dist_pred * mask.to(dist_pred.device)

        return dist_pred


def create_difference_matrix(rows: int, cols: int) -> torch.Tensor:
    row_idx = torch.arange(rows).unsqueeze(1).expand(-1, cols)
    col_idx = torch.arange(cols).unsqueeze(0).expand(rows, -1)
    return torch.abs(row_idx - col_idx).float()


def compute_r2_score_with_mask(y_true: torch.Tensor, y_pred: torch.Tensor, lengths: List[int]):
    B, L, _ = y_true.shape
    mask_list = []
    for l in lengths:
        m = torch.zeros(L, L)
        m[1:l+1, 1:l+1] = 1
        mask_list.append(m)
    mask = torch.stack(mask_list).bool().to(y_true.device)
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    r2 = 1 - (torch.sum((y_true_masked - y_pred_masked) ** 2) /
              torch.sum((y_true_masked - y_true_masked.mean()) ** 2))
    return float(r2)


def load_model_local_only(local_dir: str):
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"Local model directory not found: {local_dir}")

    required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
    for f in required_files:
        path = os.path.join(local_dir, f)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required file {f} not found in {local_dir}")

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModel.from_pretrained(local_dir)

    return tokenizer, model

    
def main():
    LEARNING_RATE = 3e-5
    epochs = 5
    TRAIN_BATCH_SIZE = 2
    VAL_BATCH_SIZE = 1
    weight_decay = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision("high")

    #model_name  = "facebook/esm2_t33_650M_UR50D"
    #hidden_size = 1280

    model_name  = "facebook/esm2_t30_150M_UR50D"
    hidden_size = 640

    local_map = {
        "facebook/esm2_t30_150M_UR50D": "./esm2_t30_150M_UR50D",
        "facebook/esm2_t33_650M_UR50D": "./esm2_t33_650M_UR50D",
    }
    local_model_dir = local_map.get(model_name, f"./local_cache_{model_name.replace('/', '_')}")

    tokenizer, esm_encoder = load_model_local_only(local_model_dir)
    esm_encoder = esm_encoder.to(device)
    print("ESM encoder loaded")

    npz_dir = "distance_maps"
    dataset = NPZSequenceDistanceDataset(npz_dir)
    sequences, distance_maps = zip(*[dataset[i] for i in range(len(dataset))])

    train_sequences, val_sequences, train_maps, val_maps = train_test_split(
        sequences, distance_maps, test_size=0.2, random_state=0
    )

    train_dataset = SequenceDistanceDataset(train_sequences, train_maps)
    val_dataset = SequenceDistanceDataset(val_sequences, val_maps)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, tokenizer))

    model = DistanceMapPredictor(esm_model=esm_encoder, hidden_size=hidden_size).to(device)
    
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=weight_decay
    )
    
    criterion = nn.MSELoss(reduction="none")
    best_r2 = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}/{epochs} - Train")

        for batch in train_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["distance_maps"].to(device)
            lengths = batch["lengths"]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, lengths)

            abs_diff = create_difference_matrix(outputs.size(1), outputs.size(2)).to(device)
            weights = abs_diff ** 0.5
            
            mask = (targets != 0).float()
            losses = criterion(outputs, targets)
            weighted_loss = (losses * weights * mask).sum() / mask.sum()

            weighted_loss.backward()
            optimizer.step()

            running_loss += weighted_loss.item()
            train_loop.set_postfix(loss=weighted_loss.item())

        print(f"Epoch {epoch+1} - Avg Train Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        r2_scores = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["distance_maps"].to(device)
                lengths = batch["lengths"]

                outputs = model(input_ids, attention_mask, lengths)

                abs_diff = create_difference_matrix(outputs.size(1), outputs.size(2)).to(device)
                weights = abs_diff ** 0.5

                mask = (targets != 0).float()
                losses = criterion(outputs, targets)
                weighted_loss = (losses * weights * mask).sum() / mask.sum()
                
                val_loss += weighted_loss.item()

                r2 = compute_r2_score_with_mask(targets, outputs, lengths)
                r2_scores.append(r2)

        avg_r2 = np.mean(r2_scores)
        print(f"Epoch {epoch+1} - Avg Val Loss: {val_loss/len(val_loader):.4f}, Avg R^2: {avg_r2:.4f}")

        if avg_r2 > best_r2:
            best_r2 = avg_r2
            torch.save(model.state_dict(), "best_distance_map_model.pth")
            print(f"Saved best model with R^2={best_r2:.4f}")

if __name__ == "__main__":
    main()
