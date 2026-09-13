"""Train the paper's SOVA model or its matched TW-GRPO control."""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainingArguments, set_seed

from .core import answer_set
from .trainer import SOVATrainer


def load_data(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Data must be a nonempty JSON array")
    for index, row in enumerate(rows):
        if not all(isinstance(row.get(key), str) and row[key] for key in ("video", "problem", "solution")):
            raise ValueError(f"Row {index} needs video, problem, and solution strings")
        if not Path(row["video"]).is_file() or not answer_set(row["solution"]):
            raise ValueError(f"Row {index} has a missing video or empty answer set")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Qwen2.5-VL checkpoint path or Hub ID")
    parser.add_argument("--data", required=True, help="Prepared CLEVRER training JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lambda-positive", type=float, default=0.0625)
    parser.add_argument("--lambda-negative", type=float, default=0.03125)
    parser.add_argument("--deepspeed", default="scripts/zero3_offload.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; use a separate directory for every run")
    rows = load_data(args.data)
    if len(rows) < 1000:
        raise ValueError("The paper protocol requires at least 1000 training queries")
    # Fix the query subset across seeds; the sampler and policy use the run seed.
    dataset = Dataset.from_list(rows).shuffle(seed=42).select(range(1000))
    set_seed(args.seed)
    training = TrainingArguments(
        output_dir=args.output, learning_rate=1e-6, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, max_steps=500, num_train_epochs=1,
        seed=args.seed, data_seed=args.seed, bf16=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=20, deepspeed=args.deepspeed, remove_unused_columns=False,
        logging_steps=1, save_steps=500, save_only_model=True, report_to="none",
        lr_scheduler_type="linear", warmup_ratio=0, weight_decay=0,
    )
    if training.world_size != 2:
        raise ValueError("The paper protocol requires two processes: torchrun --nproc_per_node=2")
    config = AutoConfig.from_pretrained(args.model)
    if config.model_type != "qwen2_5_vl":
        raise ValueError("This minimal release supports Qwen2.5-VL-3B/7B")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", use_cache=False)
    for name, parameter in model.named_parameters():
        if name.startswith("visual."):
            parameter.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=128 * 28 * 28, min_pixels=4 * 28 * 28)
    processor.tokenizer.padding_side = "left"
    trainer = SOVATrainer(
        model=model, args=training, train_dataset=dataset, processing_class=processor,
        data_collator=lambda rows: rows, group_size=args.group_size,
        lambda_positive=args.lambda_positive, lambda_negative=args.lambda_negative,
    )
    trainer.train()
    trainer.save_model(args.output)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output)
        (output / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
