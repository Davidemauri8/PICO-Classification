"""BERT fine-tuning for binary PICO-element classification.

Mirrors the structure of train_tpot.py and train_classification.py:
CLI args, shared data_utils for loading/undersampling, a single seed used
everywhere, and a held-out test set that is evaluated exactly once, at the
end, after model selection on the validation set.
"""

import argparse
import time
import datetime
from pathlib import Path

import numpy as np
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)

from data_utils import load_dataset, underSample2Min


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dati_Pico.xlsx")
    parser.add_argument("--seed", type=int, default=1702)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--val-size", type=float, default=0.15,
                         help="fraction of the non-test data used for validation")
    parser.add_argument("--tokenizer", default="bert-base-uncased")
    parser.add_argument("--out", default="results/reports")
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=180):
        self.texts = dataframe["Text"].values
        self.targets = dataframe["Category"].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.as_tensor(self.targets[idx], dtype=torch.long),
        }


def run_epoch(model, loader, device, optimizer=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates.

    Loss and predictions are accumulated fresh for this single epoch/pass
    only (no cross-epoch leakage), and evaluation is always mini-batched.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses, all_preds, all_targets = [], [], []
    for batch in loader:
        targets = batch["targets"].to(device)
        mask = batch["attention_mask"].to(device)
        ids = batch["input_ids"].to(device)

        if is_train:
            model.zero_grad()
            loss, logits = model(
                ids, token_type_ids=None, attention_mask=mask, labels=targets
            ).to_tuple()
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, logits = model(
                    ids, token_type_ids=None, attention_mask=mask, labels=targets
                ).to_tuple()

        losses.append(loss.item())
        all_preds.extend(np.argmax(logits.cpu().detach().numpy(), axis=1).flatten())
        all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="binary"
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }, all_preds, all_targets


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    df = load_dataset(args.data)
    df = underSample2Min(df, "Category", random_state=args.seed)

    # Three-way split, single seed, stratified: test is set aside first and
    # touched exactly once, at the very end.
    train_val, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed,
        stratify=df["Category"],
    )
    train_df, val_df = train_test_split(
        train_val, test_size=args.val_size, random_state=args.seed,
        stratify=train_val["Category"],
    )
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_dataset = TextDataset(train_df, tokenizer, args.max_length)
    val_dataset = TextDataset(val_df, tokenizer, args.max_length)
    test_dataset = TextDataset(test_df, tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    torch.cuda.empty_cache()
    device = "cuda" if cuda.is_available() else "cpu"

    model = BertForSequenceClassification.from_pretrained(
        args.tokenizer, num_labels=2,
        output_attentions=False, output_hidden_states=False,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    best_val_f1 = -1.0
    best_state = None
    training_stats = []
    total_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n================ Epoch {epoch} / {args.epochs} ================")

        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer)
        elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
        print(f"---TRAIN METRICS--- (elapsed {elapsed})")
        for k, v in train_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        print("\nRunning validation ...")
        val_metrics, _, _ = run_epoch(model, val_loader, device, optimizer=None)
        print("---VALIDATION METRICS---")
        for k, v in val_metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"New best validation F1: {best_val_f1:.4f} (checkpoint kept)")

        training_stats.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

    print(f"\nTotal training time: "
          f"{str(datetime.timedelta(seconds=int(round(time.time() - total_t0))))}")

    # Final evaluation, once, on the held-out test set — using the checkpoint
    # with the best validation F1, never used for model selection until now.
    if best_state is not None:
        model.load_state_dict(best_state)
    _, test_preds, test_targets = run_epoch(model, test_loader, device, optimizer=None)

    report = classification_report(test_targets, test_preds, digits=4)
    print("\n---TEST METRICS (best checkpoint, held-out set)---")
    print(report)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bert.txt").write_text(report)


if __name__ == "__main__":
    main()
