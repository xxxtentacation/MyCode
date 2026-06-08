from __future__ import annotations

import argparse

import torch

from src.dataset import preprocess_text, tokenize
from src.model import create_model


def encode(text: str, token_to_id, use_preprocess: bool):
    encoded_text = preprocess_text(text) if use_preprocess else text
    tokens = tokenize(encoded_text)
    unk = token_to_id.get("<unk>", 1)
    ids = [token_to_id.get(tok, unk) for tok in tokens] or [unk]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float32)
    return input_ids, attention_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict robotic-arm action label from a text command.")
    parser.add_argument("--checkpoint", default="artifacts/model_compare/best.pt")
    parser.add_argument("--text", required=True)
    parser.add_argument("--disable-preprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    token_to_id = ckpt["token_to_id"]
    label_to_id = ckpt["label_to_id"]
    id_to_label = {v: k for k, v in label_to_id.items()}
    use_preprocess = ckpt.get("use_preprocess", True) and (not args.disable_preprocess)

    model_type = ckpt.get("model_type", "tiny")
    model = create_model(
        model_type=model_type,
        vocab_size=len(token_to_id),
        num_labels=len(label_to_id),
        embed_dim=ckpt["embed_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    input_ids, attention_mask = encode(args.text, token_to_id, use_preprocess=use_preprocess)
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        pred_id = int(logits.argmax(dim=1).item())

    print(id_to_label[pred_id])


if __name__ == "__main__":
    main()

