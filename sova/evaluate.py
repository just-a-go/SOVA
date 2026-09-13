"""Evaluate multiple-choice video QA with the paper's decoding settings."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

from .core import answer_set, score_response
from .train import load_data
from .video import messages, read_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--benchmark", choices=["clevrer", "nextgqa", "mmvu", "mvbench", "tempcompass", "videomme"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; choose a new filename")
    rows = load_data(args.data)
    if any(row.get("problem_type", "multiple choice") != "multiple choice" for row in rows):
        raise ValueError("Use the multiple-choice benchmark subsets reported in the paper")
    set_seed(args.seed)
    if AutoConfig.from_pretrained(args.model).model_type != "qwen2_5_vl":
        raise ValueError("This minimal release supports Qwen2.5-VL-3B/7B")
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=256 * 28 * 28, min_pixels=4 * 28 * 28)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto").eval()
    results = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for start in tqdm(range(0, len(rows), args.batch_size)):
        batch = rows[start:start + args.batch_size]
        text = [processor.apply_chat_template(messages(row, evaluation=True, general=args.benchmark != "clevrer"),
                                             tokenize=False, add_generation_prompt=True) for row in batch]
        videos = [read_video(row["video"], 256 * 28 * 28) for row in batch]
        inputs = processor(text=text, videos=videos, padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=4096, do_sample=True,
                                       temperature=0.01, top_p=0.001, top_k=50,
                                       pad_token_id=processor.tokenizer.pad_token_id)
        responses = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for row, response in zip(batch, responses):
            accuracy, formatting = score_response(response, row["solution"])
            results.append({"video": row["video"], "problem": row["problem"], "solution": row["solution"],
                            "response": response, "correct": answer_set(response) == answer_set(row["solution"]),
                            "partial_credit": accuracy, "format": formatting})
    metrics = {"samples": len(results), "strict_accuracy": 100 * sum(row["correct"] for row in results) / len(results),
               "partial_credit": 100 * sum(row["partial_credit"] for row in results) / len(results),
               "format_compliance": 100 * sum(row["format"] for row in results) / len(results)}
    output.write_text(json.dumps({"config": vars(args), "metrics": metrics, "results": results}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
