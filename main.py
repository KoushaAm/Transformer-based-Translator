import argparse
import os
import json
import torch
from enum import Enum
from datasets import load_dataset
from train import train as train_lora_model
from contrastive import contrastive_decode
from contrastive_search import contrastive_search_decode
from evaluate import compute_bleu
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dictionary import DictionaryHelper
from peft import PeftModel
import glob


TARGET_LANG = "Cantonese" # old: Mandarin
TARGET_LANG_CODE = "yue" # old: zh
MODEL_NAME = "google/mt5-small"


class DecodeMode(Enum):
    BEAM_HINT = "beam_hint"
    BEAM_BASE = "beam_base"
    GREEDY_CONTRASTIVE = "greedy_contrastive"
    GREEDY_HINT = "greedy_hint"
    GREEDY_BASE = "greedy_base"


# Data processing functions
def split_dataset_by_ratio(ds, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"
    train_test = ds.train_test_split(train_size=int(len(ds) * train_ratio), seed=seed)
    remaining = train_test["test"]
    val_test = remaining.train_test_split(
        train_size=int(len(ds) * val_ratio), seed=seed
    )
    return {
        "train": train_test["train"],
        "val": val_test["train"],
        "test": val_test["test"],
    }


def convert_to_json(dataset, lang_pair, output_path):
    src_lang, tgt_lang = lang_pair
    data = [
        {
            "src": ex.get("translation", ex)[src_lang],
            "tgt": ex.get("translation", ex)[tgt_lang],
        }
        for ex in dataset
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} examples to {output_path}")


def prepare_dataset(args):
    paths = [
        f"data/en_{TARGET_LANG_CODE}_{split}.json" for split in ["train", "val", "test"]
    ]
    if not args.prepare_data and all(os.path.exists(p) for p in paths):
        print("Dataset already prepared.")
        return

    print("Loading dataset from HuggingFace...")

    if TARGET_LANG_CODE == "yue":
        print("Loading dataset from HuggingFace: lordjia/Cantonese_English_Translation")
        # single 'train' split on HF; we split it ourselves
        raw_ds = load_dataset("lordjia/Cantonese_English_Translation", split="train")
        raw_ds = raw_ds.shuffle(seed=42)

        # capping dataset size (comment out if you want full 110k)
        raw_ds = raw_ds.select(range(min(125000, len(raw_ds))))

        print(f"Total examples: {len(raw_ds)}")

        splits = split_dataset_by_ratio(raw_ds)
        os.makedirs("data", exist_ok=True)

        lang_pair = ("english", "cantonese")
        for name, split in splits.items():
            convert_to_json(
                split,
                lang_pair,
                f"data/en_{TARGET_LANG_CODE}_{name}.json",
            )
    else :
        raw_ds = load_dataset("wmt18", f"{TARGET_LANG_CODE}-en")["train"].shuffle(seed=42)
        raw_ds = raw_ds.select(range(min(100000, len(raw_ds))))
        print(f"Total examples: {len(raw_ds)}")

        splits = split_dataset_by_ratio(raw_ds)
        os.makedirs("data", exist_ok=True)

        lang_pair = ("en", TARGET_LANG_CODE)
        for name, split in splits.items():
            convert_to_json(split, lang_pair, f"data/en_{TARGET_LANG_CODE}_{name}.json")

    print("Dataset preparation complete.")


# inference
def load_models_experiment(device: str):
    #load tokenizer, hint (strong) model and base (weak) model as available.
    #returns (tokenizer, hint_model, base_model). if a Peft checkpoint exists in
    #hint_model or model it will be loaded. otherwise falls back to
    #AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME) for that role.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    hint_model = None
    base_model = None

    if os.path.exists("hint_model"):
        print("Loading hint (strong) model from 'hint_model'...")

        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        hint_model = PeftModel.from_pretrained(base, "hint_model").to(device).eval()
    else:
        print("No 'hint_model' directory found; hint model not loaded.")

    if os.path.exists("model"):
        print("Loading base (weak) model from 'model'...")
        base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        base_model = PeftModel.from_pretrained(base, "model").to(device).eval()
    else:
        print("No 'model' directory found; base model not loaded.")

    return tokenizer, hint_model, base_model

def generate_prediction(mode: DecodeMode, tokenizer, hint_model, base_model, prompt, device, weak_prompt=None):
    max_new_tokens = 50
    repetition_penalty = 1.1

    if mode == DecodeMode.BEAM_HINT:
        if hint_model is None:
            raise ValueError("Hint model is not loaded for BEAM_HINT")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = hint_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=8,
                length_penalty=1.0,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    if mode == DecodeMode.BEAM_BASE:
        if base_model is None:
            raise ValueError("Base model is not loaded for BEAM_BASE")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=8,
                length_penalty=1.0,
                repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    if mode == DecodeMode.GREEDY_CONTRASTIVE:
        if hint_model is None or base_model is None:
            raise ValueError("Both hint and base models must be loaded for GREEDY_CONTRASTIVE")
        # prompt is the strong (hinted) prompt; weak_prompt is the base prompt
        return contrastive_decode(
            hint_model,
            base_model,
            tokenizer,
            prompt,
            weak_prompt=weak_prompt,
            max_new_tokens=max_new_tokens,
            alpha=0.3,
            top_k=5,
            repetition_penalty=repetition_penalty,
            do_sample=False,
        )

    if mode == DecodeMode.GREEDY_HINT:
        if hint_model is None:
            raise ValueError("Hint model is not loaded for GREEDY_HINT")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = hint_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    # GREEDY_BASE
    if mode == DecodeMode.GREEDY_BASE:
        if base_model is None:
            raise ValueError("Base model is not loaded for GREEDY_BASE")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(gen[0], skip_special_tokens=True)

    raise ValueError(f"Unsupported decode mode: {mode}")


def run_inference(
        mode: DecodeMode, decode_small: bool = False, retrieve_vocab: bool = False, max_examples: int = 1000
):
    output_file = f"output/inference_results_{mode.value}.json"

    if retrieve_vocab:
        output_file = f"output/inference_results_with_hints_{mode.value}.json"

    print(f"Running inference with mode: {mode.value}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer, hint_model, base_model = load_models_experiment(device)

    with open(f"data/en_{TARGET_LANG_CODE}_test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if max_examples:
        test_data = test_data[:max_examples]
    if decode_small:
        test_data = test_data[:100]

    dictionary = DictionaryHelper()

    print(f"Running inference on {len(test_data)} examples...")

    outputs, refs = [], {}
    for i, ex in enumerate(test_data):
        src, tgt = ex["src"], ex["tgt"]
        refs[i] = [tgt]

        base_prompt = f"translate English to {TARGET_LANG}: {src}"

        hints = dictionary.get_hints(src)
        hint_prompt = f"{hints}{base_prompt}"

        if mode == DecodeMode.GREEDY_CONTRASTIVE:
            final_prompt = hint_prompt
            weak_prompt_arg = base_prompt
        elif mode in (DecodeMode.BEAM_HINT, DecodeMode.GREEDY_HINT) or retrieve_vocab:
            final_prompt = hint_prompt
            weak_prompt_arg = None
        else:
            # BEAM_BASE or GREEDY_BASE
            final_prompt = base_prompt
            weak_prompt_arg = None

        pred = generate_prediction(
            mode, tokenizer, hint_model, base_model, final_prompt, device, weak_prompt=weak_prompt_arg
        ).strip()

        outputs.append({"index": i, "src": src, "tgt": tgt, "pred": pred})

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_data)}")
        if i < 5:
            print(
                f"\n--- Example ({mode.value}) ---\nSRC: {src}\nREF: {tgt}\nPRED: {pred}"
            )

    # w:write outputs to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Inference results saved to {output_file}")

    bleu = compute_bleu(refs, [(o["index"], o["pred"]) for o in outputs])
    print(f"\n{mode.value} BLEU: {bleu:.2f}")
    return bleu, outputs, refs


def evaluate_from_file(input_file: str):
    # Evaluate predictions from a file
    print(f"Evaluating predictions from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        outputs = json.load(f)

    refs = {o["index"]: [o["tgt"]] for o in outputs}
    predictions = [(o["index"], o["pred"]) for o in outputs]

    bleu = compute_bleu(refs, predictions)
    print(f"Evaluation BLEU: {bleu:.2f}")
    return bleu


def evaluate_all():
    output_dir = "output"
    json_files = glob.glob(f"{output_dir}/*.json")

    if not json_files:
        print(f"No JSON files found in {output_dir} to evaluate.")
        return

    print(f"Found {len(json_files)} JSON files in {output_dir} to evaluate.")

    for json_file in json_files:
        try:
            print(f"\nEvaluating {json_file}...")
            evaluate_from_file(json_file)
        except Exception as e:
            print(f"Error evaluating {json_file}: {e}")


# cli
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--skip-train", action="store_true")
    parser.add_argument("-b", "--base-model", action="store_true")
    parser.add_argument("-l", "--lora-only", action="store_true")
    parser.add_argument("--contrastive-search", action="store_true")
    parser.add_argument("-s", "--decode-small", action="store_true")
    parser.add_argument("-p", "--prepare-data", action="store_true")
    parser.add_argument("-r", "--retrieve-vocab", action="store_true")
    parser.add_argument("--scenario", type=int, choices=[1,2,3,4,5], help="Experiment scenario number: 1=beam+hint, 2=beam+base, 3=greedy_contrastive, 4=greedy_hint, 5=greedy_base")
    parser.add_argument("--evaluate-only", type=str, help="Path to file for evaluation only")
    parser.add_argument("-e", "--evaluate-all", action="store_true", help="Evaluate all JSON files in the output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.evaluate_all:
        evaluate_all()
        return

    if args.evaluate_only:
        evaluate_from_file(args.evaluate_only)
        return

    print("=" * 60 + "\n### Preparing dataset...\n" + "=" * 60)
    prepare_dataset(args)
    if args.prepare_data:
        return

    # choose decode mode
    if args.scenario:
        mapping = {
            1: DecodeMode.BEAM_HINT,
            2: DecodeMode.BEAM_BASE,
            3: DecodeMode.GREEDY_CONTRASTIVE,
            4: DecodeMode.GREEDY_HINT,
            5: DecodeMode.GREEDY_BASE,
        }
        mode = mapping[args.scenario]
    else:
        # default to greedy contrastive (closest to previous default)
        mode = DecodeMode.GREEDY_CONTRASTIVE

    if not args.skip_train and mode not in (DecodeMode.GREEDY_BASE, DecodeMode.BEAM_BASE):
        print("\n" + "=" * 60 + "\n### Training LoRA model...\n" + "=" * 60)
        train_lora_model()
    else:
        print("\n" + "=" * 60 + "\n### Skipping training\n" + "=" * 60)

    print("\n" + "=" * 60 + f"\n### Decoding: {mode.value}\n" + "=" * 60)
    run_inference(mode, decode_small=args.decode_small, retrieve_vocab=args.retrieve_vocab)


if __name__ == "__main__":
    main()
