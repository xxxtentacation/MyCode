from __future__ import annotations

import argparse
from collections import Counter
import json

import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

from src.dataset import VocabBundle, TextCommandDataset, collate_batch, read_labeled_text
from src.model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained tiny robotic-arm command classifier.")
    parser.add_argument("--checkpoint", default="artifacts/model_compare/best.pt")
    parser.add_argument("--test", default="data/test.txt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--disable-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    token_to_id = ckpt["token_to_id"]
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    use_preprocess = ckpt.get("use_preprocess", True) and (not args.disable_preprocess)

    test_samples = read_labeled_text(args.test)

    vocab = VocabBundle(token_to_id=token_to_id, label_to_id=label_to_id)
    test_ds = TextCommandDataset(test_samples, vocab, use_preprocess=use_preprocess)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model_type = ckpt.get("model_type", "tiny")
    model = create_model(
        model_type=model_type,
        vocab_size=len(token_to_id),
        num_labels=len(label_to_id),
        embed_dim=ckpt["embed_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            pred = logits.argmax(dim=1)
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(pred.tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"accuracy: {acc:.3f}")
    print(f"macro_f1: {macro_f1:.3f}")
    print("class distribution (true):", dict(Counter(id_to_label[i] for i in y_true)))
    print("\nclassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=sorted(id_to_label.keys()),
            target_names=[id_to_label[i] for i in sorted(id_to_label.keys())],
            digits=3,
            zero_division=0,
        )
    )

    if args.metrics_out:
        report = classification_report(
            y_true,
            y_pred,
            labels=sorted(id_to_label.keys()),
            target_names=[id_to_label[i] for i in sorted(id_to_label.keys())],
            digits=3,
            zero_division=0,
            output_dict=True,
        )
        payload = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "class_distribution": dict(Counter(id_to_label[i] for i in y_true)),
            "report": report,
        }
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"metrics json saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()

