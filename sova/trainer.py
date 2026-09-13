# Adapted from TW-GRPO and Hugging Face TRL (Apache-2.0).
"""One query per device, one update per freshly sampled response group."""

from collections import defaultdict

import torch
from transformers import GenerationConfig, Trainer
from trl.models import unwrap_model_for_generation

from .core import advantages, completion_mask, policy_loss, score_response
from .video import messages, read_video


class SOVATrainer(Trainer):
    def __init__(self, *args, group_size=8, lambda_positive=0.0625,
                 lambda_negative=0.03125, alpha=0.7, max_length=4096, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.per_device_train_batch_size != 1:
            raise ValueError("The paper protocol uses one query per device")
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        # Validate before the first expensive policy rollout.
        advantages(torch.zeros(1, group_size), torch.zeros(1, group_size), lambda_positive, lambda_negative)
        self.group_size = group_size
        self.lambda_positive, self.lambda_negative = lambda_positive, lambda_negative
        self.alpha, self.max_length = alpha, max_length
        self.model_accepts_loss_kwargs = False
        self._metrics = defaultdict(list)
        self.generation = GenerationConfig(
            max_new_tokens=max_length, do_sample=True, temperature=1.0,
            top_p=1.0, top_k=50, num_return_sequences=group_size,
            pad_token_id=self.processing_class.tokenizer.pad_token_id,
            eos_token_id=self.processing_class.tokenizer.eos_token_id,
        )

    def _prepare_inputs(self, inputs):
        return inputs  # Raw video paths are processed inside compute_loss.

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs or len(inputs) != 1:
            raise ValueError("Use the separate evaluator; training expects one query per device")
        row = inputs[0]
        processor = self.processing_class
        text = processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)
        video = read_video(row["video"], 128 * 28 * 28)
        prompt = processor(text=[text], videos=[video], padding=True, return_tensors="pt")
        prompt = {key: value.to(self.accelerator.device) if isinstance(value, torch.Tensor) else value
                  for key, value in prompt.items()}
        prompt_length = prompt["input_ids"].shape[1]
        if prompt_length > self.max_length:
            raise ValueError("Prompt exceeds the token limit; shorten the question without truncating video tokens")

        with unwrap_model_for_generation(model, self.accelerator) as policy:
            with torch.no_grad():
                sequences = policy.generate(**prompt, generation_config=self.generation, use_cache=True)
        completion = sequences[:, prompt_length:]
        mask = completion_mask(completion, processor.tokenizer.eos_token_id)
        responses = processor.batch_decode(completion, skip_special_tokens=True)
        scores = torch.tensor([score_response(text, row["solution"]) for text in responses],
                              dtype=torch.float32, device=sequences.device)
        calibrated, baseline, positive, negative = advantages(
            scores[:, 0][None], scores[:, 1][None], self.lambda_positive, self.lambda_negative)

        # There is one video/query on this rank. Generate returns G contiguous responses.
        forward_inputs = {
            "input_ids": sequences,
            "attention_mask": torch.cat((prompt["attention_mask"].repeat(self.group_size, 1), mask.long()), dim=1),
            "use_cache": False,
        }
        for key in ("pixel_values_videos", "video_grid_thw"):
            forward_inputs[key] = prompt[key].repeat(self.group_size, 1)
        if "second_per_grid_ts" in prompt:
            value = prompt["second_per_grid_ts"]
            forward_inputs["second_per_grid_ts"] = value.repeat(self.group_size) if isinstance(value, torch.Tensor) else value * self.group_size

        logits = model(**forward_inputs).logits[:, prompt_length - 1:-1]
        loss = policy_loss(logits, completion, mask, calibrated, self.group_size, self.alpha)
        metrics = {
            "reward/accuracy": scores[:, 0].mean(), "reward/format": scores[:, 1].mean(),
            "gate/positive": positive.float().mean(), "gate/negative": negative.float().mean(),
            "gate/negative_mixed_format": (negative & (scores[:, 1].std() > 0)).float().mean(),
            "sova/correction": (calibrated - baseline).abs().mean(),
            "completion_length": mask.sum(1).mean(),
        }
        for name, value in metrics.items():
            self._metrics[name].append(self.accelerator.gather(value.detach().reshape(1)).mean().item())
        return loss

    def log(self, logs, start_time=None):
        metrics = {key: sum(values) / len(values) for key, values in self._metrics.items() if values}
        super().log({**logs, **metrics}, start_time)
        self._metrics.clear()
