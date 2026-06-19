import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from focal_loss import FocalLoss, batch_hard_triplet_loss

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class Row:
    text: str
    label: int

class GuardrailDataset(Dataset):
    def __init__(self, path: str):
        self.rows = []
        p = Path(path)
        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        text = obj.get("text") or obj.get("conversation_text") or ""
                        label = int(obj["label"])
                        if text.strip():
                            self.rows.append(Row(text=text, label=label))
        elif p.suffix == ".csv":
            with p.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    text = row.get("text") or row.get("conversation_text") or ""
                    if text.strip():
                        self.rows.append(Row(text=text, label=int(row["label"])))
        else:
            raise ValueError("Use .jsonl or .csv")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return row.text, row.label

class MiniLMGuardrail(nn.Module):
    def __init__(self, base_name=MODEL_NAME, num_labels=9, proj_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        hidden = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        emb = self.projector(pooled)
        logits = self.classifier(emb)
        return logits, emb

def collate(batch, tokenizer, max_length=256):
    texts, labels = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = torch.tensor(labels, dtype=torch.long)
    return enc

@torch.no_grad()
def compute_mahalanobis_stats(model, loader, device, out_path: Path):
    model.eval()
    embs = []
    labels = []

    for batch in loader:
        y = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        _, emb = model(batch["input_ids"], batch["attention_mask"])
        embs.append(emb.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    X = np.concatenate(embs, axis=0)
    y = np.concatenate(labels, axis=0)

    mean = X.mean(axis=0)
    cov = np.cov(X.T)
    cov += np.eye(cov.shape[0], dtype=np.float32) * 1e-3
    inv_cov = np.linalg.pinv(cov).astype(np.float32)

    np.savez(out_path, mean=mean.astype(np.float32), inv_cov=inv_cov, labels=y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-labels", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = MiniLMGuardrail(num_labels=args.num_labels)

    train_ds = GuardrailDataset(args.train)
    val_ds = GuardrailDataset(args.val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok, args.max_length)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate(b, tok, args.max_length)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    focal = FocalLoss(gamma=2.0)

    best_acc = 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            opt.zero_grad()
            logits, emb = model(batch["input_ids"], batch["attention_mask"])
            loss_cls = focal(logits, labels)
            loss_tri = batch_hard_triplet_loss(emb, labels, margin=0.3)
            loss = loss_cls + 0.2 * loss_tri

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                logits, _ = model(batch["input_ids"], batch["attention_mask"])
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / max(1, total)
        print(f"epoch={epoch+1} train_loss={train_loss / max(1, len(train_loader)):.4f} val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_dir / "minilm_guardrail.pt")
            tok.save_pretrained(out_dir)

    print(f"best_val_acc={best_acc:.4f}")
    compute_mahalanobis_stats(model, train_loader, device, out_dir / "mahalanobis_stats.npz")

if __name__ == "__main__":
    main()