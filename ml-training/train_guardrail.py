import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from focal_loss import FocalLoss


@dataclass
class Sample:
    text: str
    label: int


class JsonlCsvDataset(Dataset):
    def __init__(self, path: str):
        self.samples: List[Sample] = []

        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self.samples.append(Sample(text=obj["text"], label=int(obj["label"])))
        elif path.endswith(".csv"):
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.samples.append(Sample(text=row["text"], label=int(row["label"])))
        else:
            raise ValueError("Use .jsonl or .csv")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length=256):
    texts = [x.text for x in batch]
    labels = torch.tensor([x.label for x in batch], dtype=torch.long)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = labels
    return enc


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    loss_fn = FocalLoss(gamma=2.0)

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loss_sum += loss.item() * labels.size(0)

    return {
        "val_loss": loss_sum / max(1, total),
        "val_acc": correct / max(1, total)
    }


def train(
    train_path: str,
    val_path: str,
    output_dir: str,
    model_name: str = "microsoft/deberta-v3-base",
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_length: int = 256
):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_ds = JsonlCsvDataset(train_path)
    val_ds = JsonlCsvDataset(val_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, total_steps // 10),
        num_training_steps=total_steps
    )
    loss_fn = FocalLoss(gamma=2.0)

    best_acc = -1.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(1, len(train_loader))

        print(
            f"epoch={epoch+1} train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_acc']:.4f}"
        )

        if metrics["val_acc"] > best_acc:
            best_acc = metrics["val_acc"]
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"best validation accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to train.jsonl or train.csv")
    parser.add_argument("--val", required=True, help="Path to val.jsonl or val.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--model", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        output_dir=args.out,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()