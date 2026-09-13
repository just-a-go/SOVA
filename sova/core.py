"""SOVA equations and the inherited TW-GRPO objective (no model dependencies)."""

import math
import re

import torch
from torch.utils.checkpoint import checkpoint


def answer_set(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = match.group(1) if match else text
    return {part.strip() for part in answer.strip().split(",") if part.strip()}


def score_response(response, solution):
    """Subset accuracy, Eq. (1), and the original TW-GRPO format verifier."""
    prediction, target = answer_set(response), answer_set(solution)
    if not target:
        raise ValueError("The ground-truth answer set must be nonempty")
    accuracy = len(prediction) / len(target) if prediction <= target else 0.0
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>[\s!]*$"
    valid_format = (
        response.count("<think>") <= 1
        and response.count("<answer>") <= 1
        and re.search(pattern, response.strip(), re.DOTALL) is not None
    )
    return accuracy, float(valid_format)


def advantages(accuracy, formatting, lambda_positive=0.0625, lambda_negative=0.03125):
    """Calibrate [queries, G] rewards; zero coefficients give the matched control.

    Anchors are fixed at 1.5 and 2.0. They enter sample statistics only.
    Supports the paper's G = 2, 4, 8, 16 study without changing the estimator.
    """
    if accuracy.ndim != 2 or accuracy.shape != formatting.shape or accuracy.shape[1] < 2:
        raise ValueError("Rewards must have matching [queries, G >= 2] shapes")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (lambda_positive, lambda_negative)):
        raise ValueError("Interpolation coefficients must be finite and in [0, 1]")
    if not torch.all(torch.isfinite(accuracy) & (accuracy >= 0) & (accuracy <= 1)):
        raise ValueError("Accuracy rewards must lie in [0, 1]")
    if not torch.all((formatting == 0) | (formatting == 1)):
        raise ValueError("Format rewards must be binary")
    rewards = (accuracy + formatting).float()
    baseline = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-4)
    positive = ((accuracy == 1) & (formatting == 1)).all(-1)
    negative = (accuracy == 0).all(-1)
    calibrated = baseline.clone()
    for gate, anchor, coefficient in ((positive, 1.5, lambda_positive), (negative, 2.0, lambda_negative)):
        if coefficient == 0:
            continue
        augmented = torch.cat((rewards, rewards.new_full((len(rewards), 1), anchor)), dim=-1)
        virtual = (rewards - augmented.mean(-1, keepdim=True)) / (augmented.std(-1, keepdim=True) + 1e-4)
        calibrated = torch.where(gate[:, None], calibrated + coefficient * (virtual - baseline), calibrated)
    return calibrated, baseline, positive, negative


def completion_mask(token_ids, eos_token_id):
    """Include the first EOS, and exclude all later padding."""
    eos = token_ids == eos_token_id
    return (eos.cumsum(-1) - eos.long() == 0).to(torch.float32)


def _divergence(log_probs, mask, group_size):
    if log_probs.ndim != 3 or log_probs.shape[:2] != mask.shape or len(mask) % group_size:
        raise ValueError("Expected [queries * G, tokens, vocabulary] log probabilities")
    grouped = log_probs.reshape(-1, group_size, *log_probs.shape[1:])
    active = mask.reshape(-1, group_size, mask.shape[-1], 1).bool()
    padded = torch.where(active, grouped, -math.log(log_probs.shape[-1]))
    difference = (padded.mean(1, keepdim=True) - padded).clamp(-11, 11)
    return (difference.exp() - difference - 1).sum(-1).mean(1)


def _weights(divergence, mask, group_size, alpha):
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    minimum, maximum = divergence.amin(-1, keepdim=True), divergence.amax(-1, keepdim=True)
    weights = 1 + alpha * (divergence - minimum) / (maximum - minimum + 1e-8)
    return weights.repeat_interleave(group_size, dim=0) * mask


def token_weights(log_probs, mask, group_size, alpha=0.7):
    """Original TW-GRPO weights in [1, 1 + alpha], including its gradient path."""
    return _weights(_divergence(log_probs, mask, group_size), mask, group_size, alpha)


def policy_loss(logits, token_ids, mask, advantage, group_size, alpha=0.7, chunk_size=32):
    """Single-use beta=0 objective; checkpoint chunks to avoid a full log-probability copy."""
    def statistics(block, ids, active):
        log_probs = block.float().log_softmax(-1)
        selected = log_probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return selected, _divergence(log_probs, active, group_size)

    chunks = [checkpoint(statistics, logits[:, start:start + chunk_size],
                         token_ids[:, start:start + chunk_size], mask[:, start:start + chunk_size],
                         use_reentrant=False)
              for start in range(0, token_ids.shape[1], chunk_size)]
    selected = torch.cat([item[0] for item in chunks], dim=-1)
    divergence = torch.cat([item[1] for item in chunks], dim=-1)
    ratio = (selected - selected.detach()).exp()
    signal = advantage.reshape(-1, 1).detach()
    surrogate = torch.minimum(ratio * signal, ratio.clamp(0.8, 1.2) * signal)
    weights = _weights(divergence, mask, group_size, alpha)
    return -(surrogate * weights * mask).sum() / mask.sum().clamp_min(1)
