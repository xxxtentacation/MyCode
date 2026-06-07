"""
Sample from a trained model, or run evaluation on the robot-instr test set.
"""
import os
import json
import pickle
from contextlib import nullcontext
import torch
import tiktoken


from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-robot-instr' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
# eval mode
eval_mode = True # if True, run evaluation on the test set instead of interactive sampling
eval_num_samples = 50 # number of test examples to evaluate (max 99)
eval_temperature = 0.0 # temperature=0 for deterministic evaluation
# -----------------------------------------------------------------------------
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', 'robot_instr', 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# =============================================================================
# Evaluation Mode
# =============================================================================
if eval_mode:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import numpy as np
    # Import the same JSON normalization used in prepare.py so expected_output
    # goes through the same conversion reversal as the training data
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'robot_instr'))

    # Configure Chinese font with fallback chain (CJK font first, DejaVu Sans for math symbols)
    _cjk_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'WenQuanYi', 'Noto Sans CJK', 'Source Han Sans']
    _cjk_fonts = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in _cjk_candidates)]
    _cjk_fonts = [f for f in _cjk_fonts if 'ExtB' not in f]  # skip SimSun-ExtB etc., incomplete glyphs
    if _cjk_fonts:
        plt.rcParams['font.sans-serif'] = [_cjk_fonts[0]] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # prevent minus sign garbled

    print(f"\n{'='*60}")
    print(f"EVALUATION MODE — Testing {eval_num_samples} samples")
    print(f"Temperature: {eval_temperature} (deterministic)" if eval_temperature == 0.0
          else f"Temperature: {eval_temperature}")
    print(f"{'='*60}\n")

    # Load evaluation data from local input.txt
    # Format: "Human: <instruction>\nRobot: <json>\n\n" repeated
    input_txt_path = os.path.join(os.path.dirname(__file__), 'data', 'robot_instr', 'input.txt')
    with open(input_txt_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Parse examples: split by double newline, extract Human/Robot pairs
    examples = []
    for block in raw_text.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            instruction = lines[0]
            expected_output = lines[1]
            if instruction.startswith("Human: ") and expected_output.startswith("Robot: "):
                instruction = instruction[len("Human: "):].strip()
                expected_output = expected_output[len("Robot: "):].strip()
                examples.append((instruction, expected_output))

    n_test = min(eval_num_samples, len(examples))
    print(f"Loaded {len(examples)} examples from {input_txt_path}, evaluating on {n_test} examples\n")

    results = []
    for i in range(n_test):
        instruction, expected_output = examples[i]

        # Build prompt: "Human: <instruction>\nRobot:"
        prompt_text = f"Human: {instruction}\nRobot:"
        prompt_ids = encode(prompt_text)
        x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

        # Generate
        with torch.no_grad():
            with ctx:
                y = model.generate(
                    x, max_new_tokens,
                    temperature=eval_temperature,
                    top_k=top_k if eval_temperature > 0 else None
                )

        # Decode full output
        full_output = decode(y[0].tolist())
        # Extract only the generated part (after the prompt)
        generated = full_output[len(prompt_text):]

        # Trim at the next example boundary (double newline) or end
        end_marker = generated.find("\n\n")
        if end_marker != -1:
            generated = generated[:end_marker]
        generated = generated.strip()

        # Parse expected JSON
        try:
            expected_json = json.loads(expected_output)
        except json.JSONDecodeError:
            expected_json = None

        # Parse generated JSON
        try:
            generated_json = json.loads(generated)
            json_valid = True
        except json.JSONDecodeError:
            generated_json = None
            json_valid = False

        # Extract function names
        expected_funcs = set()
        if expected_json:
            for cmd in expected_json:
                expected_funcs.add(cmd.get("function", ""))

        generated_funcs = set()
        if generated_json:
            for cmd in generated_json:
                generated_funcs.add(cmd.get("function", ""))

        # Metrics
        # Normalize expected_output through json.loads → json.dumps for fair comparison
        # (no reverse_conversions needed — input.txt was already processed by format_example)
        try:
            expected_parsed = json.loads(expected_output)
            expected_normalized = json.dumps(expected_parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            expected_normalized = expected_output

        # Also normalize generated through json.dumps to remove formatting differences
        try:
            gen_parsed = json.loads(generated)
            gen_normalized = json.dumps(gen_parsed, ensure_ascii=False)
        except json.JSONDecodeError:
            gen_normalized = generated

        exact_match = (gen_normalized == expected_normalized)
        func_match = (expected_funcs == generated_funcs) if expected_funcs else False
        func_overlap = len(expected_funcs & generated_funcs) / max(len(expected_funcs | generated_funcs), 1)

        # Character error rate (against normalized expected for fair comparison)
        min_len = min(len(gen_normalized), len(expected_normalized))
        max_len = max(len(gen_normalized), len(expected_normalized))
        char_errors = sum(1 for j in range(min_len) if gen_normalized[j] != expected_normalized[j])
        char_errors += (max_len - min_len)  # length difference as errors
        char_error_rate = char_errors / max_len if max_len > 0 else 0.0

        # --- Real-time output: print prediction vs actual ---
        status = "✓" if exact_match else ("~" if json_valid else "✗")
        exp_func_str = ",".join(sorted(expected_funcs))[:18] if expected_funcs else "?"
        gen_func_str = ",".join(sorted(generated_funcs))[:18] if generated_funcs else "INVALID"
        print(f"\n{'─'*70}")
        print(f"[{i}/{n_test}] {status} | {instruction}")
        print(f"  Expected : {expected_normalized[:150]}")
        print(f"  Generated: {gen_normalized[:150]}")
        if exact_match:
            print(f"  Result: EXACT MATCH ✓")
        elif json_valid:
            print(f"  Result: JSON valid, funcs={exp_func_str} → {gen_func_str}, CER={char_error_rate*100:.1f}%")
        else:
            print(f"  Result: INVALID JSON ✗, raw={generated[:100]}")
        print(f"{'─'*70}")

        results.append({
            "idx": i,
            "instruction": instruction,
            "expected": expected_normalized,
            "generated": gen_normalized,
            "json_valid": json_valid,
            "exact_match": exact_match,
            "func_match": func_match,
            "func_overlap": func_overlap,
            "char_error_rate": char_error_rate,
            "expected_funcs": expected_funcs,
            "generated_funcs": generated_funcs,
        })

    # -------------------------------------------------------------------------
    # Compute aggregate metrics
    # -------------------------------------------------------------------------
    n = len(results)
    json_valid_count = sum(1 for r in results if r["json_valid"])
    exact_match_count = sum(1 for r in results if r["exact_match"])
    func_match_count = sum(1 for r in results if r["func_match"])
    avg_char_error = np.mean([r["char_error_rate"] for r in results])
    avg_func_overlap = np.mean([r["func_overlap"] for r in results])

    json_valid_rate = json_valid_count / n * 100
    exact_match_rate = exact_match_count / n * 100
    func_match_rate = func_match_count / n * 100
    char_accuracy = (1 - avg_char_error) * 100

    # -------------------------------------------------------------------------
    # Print Summary Table
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY (n={n})")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'Value':>15}")
    print(f"{'-'*35} {'-'*15}")
    print(f"{'JSON Valid Rate':<35} {json_valid_rate:>14.1f}%")
    print(f"{'Exact Match Rate':<35} {exact_match_rate:>14.1f}%")
    print(f"{'Function Match Rate':<35} {func_match_rate:>14.1f}%")
    print(f"{'Avg Function Overlap (Jaccard)':<35} {avg_func_overlap:>14.2%}")
    print(f"{'Character-level Accuracy':<35} {char_accuracy:>14.1f}%")
    print(f"{'='*70}")

    # -------------------------------------------------------------------------
    # Print Per-Sample Table
    # -------------------------------------------------------------------------
    print(f"\n{'='*120}")
    print(f"PER-SAMPLE RESULTS")
    print(f"{'='*120}")
    header = f"{'#':<4} {'Instruction':<35} {'Valid':<6} {'Func':<6} {'Exact':<6} {'CER%':>7}"
    print(header)
    print(f"{'-'*4} {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        instr_short = r["instruction"][:32] + "..." if len(r["instruction"]) > 35 else r["instruction"]
        v = "✓" if r["json_valid"] else "✗"
        f = "✓" if r["func_match"] else "✗"
        e = "✓" if r["exact_match"] else "✗"
        print(f"{r['idx']:<4} {instr_short:<35} {v:<6} {f:<6} {e:<6} {r['char_error_rate']*100:>6.1f}%")

    # -------------------------------------------------------------------------
    # Print some failure examples
    # -------------------------------------------------------------------------
    failures = [r for r in results if not r["exact_match"]]
    if failures:
        print(f"\n{'='*60}")
        print(f"FAILURE EXAMPLES (first 3)")
        print(f"{'='*60}")
        for r in failures[:3]:
            print(f"\n--- Example {r['idx']} ---")
            print(f"Instruction: {r['instruction']}")
            print(f"Expected:     {r['expected'][:120]}")
            print(f"Generated:    {r['generated'][:120]}")
            print(f"  JSON valid={r['json_valid']}, func={r['expected_funcs']}→{r['generated_funcs']}, CER={r['char_error_rate']*100:.1f}%")

    # -------------------------------------------------------------------------
    # Save Results Chart
    # -------------------------------------------------------------------------
    chart_path = os.path.join(out_dir, "eval_results.png")
    print(f"\nGenerating evaluation chart → {chart_path}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Robot-Instruction Model Evaluation (n={n}, T={eval_temperature})',
                 fontsize=14, fontweight='bold')

    # --- Left: Metrics Bar Chart ---
    ax1 = axes[0]
    metric_names = [
        "JSON\nValid",
        "Exact\nMatch",
        "Function\nMatch",
        "Char-level\nAccuracy",
    ]
    metric_values = [json_valid_rate, exact_match_rate, func_match_rate, char_accuracy]
    colors = ['#2ecc71' if v >= 50 else '#e74c3c' for v in metric_values]

    bars = ax1.bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=1.2)
    ax1.set_ylabel('Rate (%)')
    ax1.set_title('Aggregate Metrics')
    ax1.set_ylim(0, 105)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, metric_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # --- Right: Per-sample Character Error Rate ---
    ax2 = axes[1]
    indices = [r["idx"] for r in results]
    cer_values = [r["char_error_rate"] * 100 for r in results]
    avg_cer = np.mean(cer_values)
    bar_colors = ['#3498db' if v < avg_cer else '#e67e22' for v in cer_values]

    ax2.bar(indices, cer_values, color=bar_colors, edgecolor='white', linewidth=0.5)
    ax2.axhline(y=avg_cer, color='red', linestyle='--', linewidth=1.5, label=f'Mean CER: {avg_cer:.1f}%')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Character Error Rate (%)')
    ax2.set_title('Per-Sample Character Error Rate')
    ax2.grid(axis='y', alpha=0.3)

    # --- Annotate best / worst / key samples ---
    best_idx = int(np.argmin(cer_values))
    worst_idx = int(np.argmax(cer_values))
    for i in [best_idx, worst_idx]:
        val = cer_values[i]
        va = 'bottom' if val < avg_cer else 'top'
        offset = 2 if val < avg_cer else -2
        ax2.annotate(f'{val:.1f}%', (indices[i], val),
                     textcoords="offset points", xytext=(0, offset),
                     ha='center', va=va, fontsize=7, fontweight='bold',
                     color=bar_colors[i])

    # --- Legend for bar colors ---
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='white', label=f'Below Avg CER (< {avg_cer:.1f}%): 表现较好'),
        Patch(facecolor='#e67e22', edgecolor='white', label=f'Above Avg CER (≥ {avg_cer:.1f}%): 表现较差'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=f'Mean CER = {avg_cer:.1f}%'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)

    # --- Annotation box: what is CER ---
    cer_explanation = (
        "CER = (逐字符差异数 + 长度差) ÷ max(len(gen), len(exp))\n"
        "- 柱高 0% = 完美匹配 (Exact Match)\n"
        "- 柱越高 → 生成质量越差\n"
        "- 横轴 = 测试样本序号 (0~49)"
    )
    ax2.text(0.98, 0.97, cer_explanation, transform=ax2.transAxes,
             fontsize=7, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Chart saved to {chart_path}")

    # Also save detailed results as JSON
    results_path = os.path.join(out_dir, "eval_results.json")
    serializable = []
    for r in results:
        serializable.append({
            "idx": r["idx"],
            "instruction": r["instruction"],
            "expected": r["expected"],
            "generated": r["generated"],
            "json_valid": r["json_valid"],
            "exact_match": r["exact_match"],
            "func_match": r["func_match"],
            "func_overlap": r["func_overlap"],
            "char_error_rate": r["char_error_rate"],
            "expected_funcs": list(r["expected_funcs"]),
            "generated_funcs": list(r["generated_funcs"]),
        })
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"Detailed results saved to {results_path}")

    import sys; sys.exit(0)

# =============================================================================
# Interactive Sampling Mode (original behaviour)
# =============================================================================

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
