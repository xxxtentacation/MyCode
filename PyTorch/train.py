from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import (
    TextCommandDataset,
    VocabBundle,
    build_vocab,
    collate_batch,
    read_labeled_text,
)
from src.model import create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def plot_training_curves(history: List[Dict[str, float]], output_dir: Path) -> None:
    """Generate train/val loss and accuracy curves for a single run."""
    epochs = [h["epoch"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_acc, "b-o", label="train_acc", markersize=3)
    ax1.plot(epochs, val_acc, "r-o", label="val_acc", markersize=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_loss, "b-o", label="train_loss", markersize=3)
    ax2.plot(epochs, val_loss, "r-o", label="val_loss", markersize=3)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "curves.png", dpi=150)
    plt.close(fig)


def plot_comparison_curves(
    history: List[Dict[str, float]],
    baseline_history: List[Dict[str, float]],
    output_dir: Path,
    label: str = "current",
    baseline_label: str = "baseline",
) -> None:
    """Generate side-by-side comparison curves against a previous run."""
    cur_epochs = [h["epoch"] for h in history]
    bl_epochs = [h["epoch"] for h in baseline_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(bl_epochs, [h["val_acc"] for h in baseline_history], "gray-o", label=f"{baseline_label} val_acc", markersize=3, alpha=0.7)
    ax1.plot(cur_epochs, [h["val_acc"] for h in history], "r-o", label=f"{label} val_acc", markersize=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Validation Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(bl_epochs, [h["val_loss"] for h in baseline_history], "gray-o", label=f"{baseline_label} val_loss", markersize=3, alpha=0.7)
    ax2.plot(cur_epochs, [h["val_loss"] for h in history], "r-o", label=f"{label} val_loss", markersize=3)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.set_title("Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "compare_curves.png", dpi=150)
    plt.close(fig)


def evaluate_on_test(
    checkpoint_path: Path,
    test_path: str,
    output_dir: Path,
    batch_size: int = 16,
) -> Dict[str, float]:
    """Evaluate best checkpoint on the test set and save metrics.json."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    token_to_id = ckpt["token_to_id"]
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    use_preprocess = ckpt.get("use_preprocess", True)

    test_samples = read_labeled_text(test_path)
    vocab = VocabBundle(token_to_id=token_to_id, label_to_id=label_to_id)
    test_ds = TextCommandDataset(test_samples, vocab, use_preprocess=use_preprocess, augment=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    model = create_model(
        model_type=ckpt.get("model_type", "improved"),
        vocab_size=len(token_to_id),
        num_labels=len(label_to_id),
        embed_dim=ckpt["embed_dim"],
        num_filters=ckpt.get("num_filters", 48),
        dropout=ckpt.get("dropout", 0.5),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        labels=sorted(id_to_label.keys()),
        target_names=[id_to_label[i] for i in sorted(id_to_label.keys())],
        digits=3, zero_division=0, output_dict=True,
    )

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "class_distribution": dict(Counter(id_to_label[i] for i in y_true)),
        "report": report,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"test accuracy: {acc:.3f}  macro_f1: {macro_f1:.3f}")
    return metrics


def plot_metric_bars(
    left_metrics: Dict[str, float],
    right_metrics: Dict[str, float],
    out_dir: Path,
    left_name: str,
    right_name: str,
) -> None:
    """Generate bar chart comparing accuracy and macro_f1 of two runs."""
    labels = ["accuracy", "macro_f1"]
    left_vals = [left_metrics[k] for k in labels]
    right_vals = [right_metrics[k] for k in labels]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar([i - width / 2 for i in x], left_vals, width=width, label=left_name)
    plt.bar([i + width / 2 for i in x], right_vals, width=width, label=right_name)
    plt.xticks(list(x), labels)
    plt.ylim(0.0, 1.0)
    plt.title("Test Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_metrics.png", dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny robotic-arm command classifier.")
    parser.add_argument("--train", default="data/train.txt")
    parser.add_argument("--val", default="data/val.txt")
    parser.add_argument("--output", default="artifacts/model_compare")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--disable-preprocess", action="store_true")
    parser.add_argument("--augment", action="store_true", help="Enable EDA-style text augmentation during training")

    parser.add_argument("--model", choices=["tiny", "improved"], default="tiny")
    parser.add_argument("--num-filters", type=int, default=48, help="Conv1d filters per kernel (improved model)")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (improved model)")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--no-scheduler", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--compare-with", default=None, help="Previous output dir to compare curves against")
    parser.add_argument("--test", default="data/test.txt", help="Test set path for post-training evaluation")
    return parser.parse_args()


def resolve_device(device_name: str, allow_cpu_fallback: bool) -> torch.device:
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_cpu_fallback:
            print("[warn] CUDA unavailable, fallback to CPU.")
            return torch.device("cpu")
        raise RuntimeError("CUDA is required but not available. Use --allow-cpu-fallback to continue on CPU.")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    use_preprocess = not args.disable_preprocess

    train_samples = read_labeled_text(args.train)
    val_samples = read_labeled_text(args.val)

    vocab = build_vocab(train_samples, min_freq=args.min_freq, use_preprocess=use_preprocess)
    train_ds = TextCommandDataset(train_samples, vocab, use_preprocess=use_preprocess, augment=args.augment)
    val_ds = TextCommandDataset(val_samples, vocab, use_preprocess=use_preprocess, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = resolve_device(args.device, args.allow_cpu_fallback)
    model = create_model(
        model_type=args.model,
        vocab_size=len(vocab.token_to_id),
        num_labels=len(vocab.label_to_id),
        embed_dim=args.embed_dim,
        num_filters=args.num_filters,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if not args.no_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"

    best_val_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item() * labels.size(0)

        train_acc = correct / max(total, 1)
        train_loss = epoch_loss / max(total, 1)
        val_metrics = evaluate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_metrics["acc"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_type": args.model,
                    "model_state": model.state_dict(),
                    "token_to_id": vocab.token_to_id,
                    "label_to_id": vocab.label_to_id,
                    "embed_dim": args.embed_dim,
                    "num_filters": args.num_filters,
                    "dropout": args.dropout,
                    "use_preprocess": use_preprocess,
                    "min_freq": args.min_freq,
                },
                best_path,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with (output_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "device": str(device),
                "model_type": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "label_smoothing": args.label_smoothing,
                "augment": args.augment,
                "num_filters": args.num_filters,
                "dropout": args.dropout,
                "use_scheduler": not args.no_scheduler,
                "embed_dim": args.embed_dim,
                "seed": args.seed,
                "min_freq": args.min_freq,
                "use_preprocess": use_preprocess,
            },
            f,
            indent=2,
        )

    # Evaluate best checkpoint on test set
    test_metrics = evaluate_on_test(best_path, args.test, output_dir, batch_size=args.batch_size)
    print(f"test metrics saved to: {output_dir / 'metrics.json'}")

    # Generate training curves
    plot_training_curves(history, output_dir)
    print(f"training curves saved to: {output_dir / 'curves.png'}")

    # Compare with previous run if requested
    if args.compare_with:
        baseline_dir = Path(args.compare_with)
        baseline_history_path = baseline_dir / "history.json"
        baseline_metrics_path = baseline_dir / "metrics.json"
        compare_dir = output_dir / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)

        if not baseline_history_path.exists():
            print(f"[warn] baseline history not found: {baseline_history_path}")
        else:
            with baseline_history_path.open("r", encoding="utf-8") as f:
                baseline_history = json.load(f)
            plot_comparison_curves(history, baseline_history, compare_dir,
                                   label=output_dir.name, baseline_label=baseline_dir.name)
            print(f"comparison curves saved to: {compare_dir / 'compare_curves.png'}")

            # Bar chart comparison using test metrics
            if baseline_metrics_path.exists():
                with baseline_metrics_path.open("r", encoding="utf-8") as f:
                    baseline_metrics = json.load(f)
                plot_metric_bars(baseline_metrics, test_metrics, compare_dir,
                                 left_name=baseline_dir.name, right_name=output_dir.name)
                print(f"comparison metrics saved to: {compare_dir / 'compare_metrics.png'}")

                # Summary delta
                delta = {
                    baseline_dir.name: {
                        "accuracy": baseline_metrics["accuracy"],
                        "macro_f1": baseline_metrics["macro_f1"],
                    },
                    output_dir.name: {
                        "accuracy": test_metrics["accuracy"],
                        "macro_f1": test_metrics["macro_f1"],
                    },
                    "delta": {
                        "accuracy": test_metrics["accuracy"] - baseline_metrics["accuracy"],
                        "macro_f1": test_metrics["macro_f1"] - baseline_metrics["macro_f1"],
                    },
                }
                with (compare_dir / "summary.json").open("w", encoding="utf-8") as f:
                    json.dump(delta, f, indent=2)
                print(f"comparison summary saved to: {compare_dir / 'summary.json'}")

    print(f"best checkpoint saved to: {best_path}")


if __name__ == "__main__":
    main()

