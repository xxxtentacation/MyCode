"""
Prepare the Studeni/robot-instructions dataset for character-level language modeling.
Converts NL-instruction → JSON-command pairs into a single text corpus,
then encodes to character-level integers.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder/decoder and vocab info.
"""
import os
import json
import pickle
import re
import math
import numpy as np
def parse_input_txt(filepath):
    """
    Parse input.txt back into example dicts (reverse of format_example).
    Returns list of {"input": instruction, "output": json_string} dicts.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    examples = []
    for block in raw_text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            instruction = lines[0]
            expected_output = lines[1]
            if instruction.startswith("Human: ") and expected_output.startswith("Robot: "):
                examples.append({
                    "input": instruction[len("Human: "):].strip(),
                    "output": expected_output[len("Robot: "):].strip(),
                })
    return examples



def round_floats(obj, decimals=2):
    """Recursively round all floats in a nested dict/list to given decimal places."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_floats(v, decimals) for v in obj]
    return obj


def format_example(ex):
    """Convert a single example to text line, keeping raw units (mm, radians),
    with all floats rounded to 2 decimal places for consistent token patterns."""
    instruction = ex["input"].strip()
    json_output = ex["output"].strip()
    try:
        parsed = json.loads(json_output)
        parsed = round_floats(parsed, decimals=2)
        json_output = json.dumps(parsed, ensure_ascii=False)
    except json.JSONDecodeError:
        pass  # keep original if parsing fails
    return f"Human: {instruction}\nRobot: {json_output}\n\n"

if __name__ == "__main__":
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")

    if os.path.exists(input_file_path):
        # Load from local input.txt instead of downloading from HuggingFace
        print(f"Found {input_file_path}, loading data from local file...")
        all_examples = parse_input_txt(input_file_path)
        print(f"Loaded {len(all_examples)} examples from {input_file_path}")
        # input.txt contains train+val concatenated; split 90/10 as a heuristic
        # (the original dataset has ~300 train, ~50 test; input.txt has all ~350)
        split_idx = int(len(all_examples) * 0.9)
        train_examples = all_examples[:split_idx]
        test_examples = all_examples[split_idx:]
        print(f"Split: {len(train_examples)} train, {len(test_examples)} test")
        train_texts = [format_example(ex) for ex in train_examples]
        test_texts  = [format_example(ex) for ex in test_examples]
    else:
        # Fallback: download from HuggingFace
        from datasets import load_dataset
        print("Loading Studeni/robot-instructions dataset from HuggingFace...")
        ds = load_dataset("Studeni/robot-instructions")
        print(f"Train examples: {len(ds['train'])}, Test examples: {len(ds['test'])}")
        train_texts = [format_example(ex) for ex in ds["train"]]
        test_texts  = [format_example(ex) for ex in ds["test"]]

    # Join all examples into one big corpus string
    train_data = "".join(train_texts)
    val_data   = "".join(test_texts)

    print(f"Train text length: {len(train_data):,} characters")
    print(f"Val   text length: {len(val_data):,} characters")

    # Save the full text corpus as input.txt (mimicking shakespeare_char convention)
    full_data = train_data + val_data
    input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(full_data)
    print(f"Saved full corpus to {input_file_path} ({len(full_data):,} characters)")

    # -----------------------------------------------------------------------------
    # 3. Character-level encoding (same approach as shakespeare_char)
    chars = sorted(list(set(train_data + val_data)))
    vocab_size = len(chars)
    print(f"Unique characters: {vocab_size}")
    print(f"Character set: {repr(''.join(chars))}")

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return ''.join([itos[i] for i in ids])

    # Encode both splits
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val   tokens: {len(val_ids):,}")

    # Save to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    out_dir = os.path.dirname(__file__)
    train_ids.tofile(os.path.join(out_dir, "train.bin"))
    val_ids.tofile(os.path.join(out_dir, "val.bin"))

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"\nSaved to {out_dir}/")
    print(f"  train.bin — {len(train_ids):,} tokens")
    print(f"  val.bin   — {len(val_ids):,} tokens")
    print(f"  meta.pkl  — vocab_size={vocab_size}")
    print("Done!")
