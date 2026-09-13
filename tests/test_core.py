import math

import pytest
import torch

from sova.core import advantages, completion_mask, policy_loss, score_response, token_weights


@pytest.mark.parametrize("group_size", [2, 4, 8, 16])
def test_strict_extremes_closed_form(group_size):
    accuracy = torch.tensor([[1.] * group_size, [0.] * group_size])
    formatting = torch.ones_like(accuracy)
    actual, baseline, positive, negative = advantages(accuracy, formatting)
    assert torch.equal(baseline, torch.zeros_like(baseline))
    assert positive.tolist() == [True, False] and negative.tolist() == [False, True]
    for index, gap, coefficient in [(0, -0.5, 0.0625), (1, 1., 0.03125)]:
        expected = coefficient * (-gap / (group_size + 1)) / (abs(gap) / math.sqrt(group_size + 1) + 1e-4)
        torch.testing.assert_close(actual[index], torch.full((group_size,), expected))


def test_mixed_format_has_negative_mean_but_preserves_order():
    accuracy = torch.zeros(1, 8)
    formatting = torch.tensor([[0., 1.] * 4])
    calibrated, _, _, gate = advantages(accuracy, formatting)
    assert gate.item() and calibrated.mean() < 0
    assert calibrated[0, 0] < calibrated[0, 1] and calibrated[0, 1] > 0


def test_partial_and_format_failed_success_bypass_exactly():
    accuracy = torch.tensor([[1., .5, 0., 0.], [1., 1., 1., 1.]])
    formatting = torch.tensor([[1., 1., 1., 1.], [1., 0., 1., 1.]])
    calibrated, baseline, positive, negative = advantages(accuracy, formatting)
    assert torch.equal(calibrated, baseline)
    assert not positive.any() and not negative.any()


def test_zero_coefficients_match_control_and_groups_are_independent():
    accuracy = torch.tensor([[1.] * 8, [0.] * 8, [.5] * 8])
    formatting = torch.ones_like(accuracy)
    control, baseline, _, _ = advantages(accuracy, formatting, 0, 0)
    assert torch.equal(control, baseline)
    combined = advantages(accuracy, formatting)[0]
    for index in range(3):
        assert torch.equal(combined[index], advantages(accuracy[index:index+1], formatting[index:index+1])[0][0])


@pytest.mark.parametrize("pred,target,expected", [
    ("<think>x</think><answer>A</answer>", "A,B", (.5, 1.)),
    ("<think>x</think><answer>A,C</answer>", "A,B", (0., 1.)),
    ("A,B", "A,B", (1., 0.)),
    ("<think>x</think><answer></answer>", "A", (0., 1.)),
    ("<think>x</think><answer>A</answer><answer>A</answer>", "A", (1., 0.)),
])
def test_reward_components(pred, target, expected):
    assert score_response(pred, target) == expected


def test_eos_mask_includes_only_first_eos():
    ids = torch.tensor([[1, 2, 9, 9, 0], [1, 2, 3, 4, 5]])
    assert completion_mask(ids, 9).tolist() == [[1., 1., 1., 0., 0.], [1.] * 5]


def test_tw_weights_and_gradients_match_original_rule():
    torch.manual_seed(42)
    logits = torch.randn(8, 5, 13, requires_grad=True)
    logs = logits.log_softmax(-1)
    mask = torch.ones(8, 5)
    mask[0, 3:] = 0
    actual = token_weights(logs, mask, 8)
    grouped = logs.reshape(1, 8, 5, 13)
    padded = torch.where(mask.reshape(1, 8, 5, 1) == 1, grouped, torch.full_like(grouped, -math.log(13)))
    diff = (padded.mean(1, keepdim=True) - padded).clamp(-11, 11)
    divergence = (diff.exp() - diff - 1).sum(-1).mean(1)
    low, high = divergence.min(-1, keepdim=True).values, divergence.max(-1, keepdim=True).values
    expected = (1 + .7 * (divergence - low) / (high - low + 1e-8)).repeat_interleave(8, 0) * mask
    torch.testing.assert_close(actual, expected)
    left = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    right = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(left, right)


def test_policy_loss_has_finite_nonzero_gradient():
    logits = torch.randn(8, 4, 11, requires_grad=True)
    ids = torch.randint(0, 11, (8, 4))
    advantage = advantages(torch.zeros(1, 8), torch.ones(1, 8))[0]
    loss = policy_loss(logits, ids, torch.ones(8, 4), advantage, 8)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize("coefficient", [-1., 1.01, float("nan")])
def test_invalid_coefficients_rejected(coefficient):
    with pytest.raises(ValueError):
        advantages(torch.ones(1, 8), torch.ones(1, 8), coefficient, 0)


def test_chunked_loss_matches_full_objective_and_gradient():
    torch.manual_seed(11)
    logits = torch.randn(8, 9, 17, requires_grad=True)
    ids = torch.randint(0, 17, (8, 9))
    mask = torch.ones(8, 9)
    mask[0, 4:] = 0
    signal = torch.randn(1, 8)
    actual = policy_loss(logits, ids, mask, signal, 8, chunk_size=3)
    logs = logits.log_softmax(-1)
    selected = logs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    ratio = (selected - selected.detach()).exp()
    expected = -(ratio * signal.reshape(-1, 1) * token_weights(logs, mask, 8) * mask).sum() / mask.sum()
    torch.testing.assert_close(actual, expected)
    left = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    right = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(left, right)


def test_clevrer_conversion_preserves_choice_order_and_filters_empty_targets(tmp_path):
    from sova.prepare_clevrer import convert
    (tmp_path / "video_00000.mp4").touch()
    choices = [{"choice_id": 1, "choice": "second", "answer": "correct"},
               {"choice_id": 0, "choice": "first", "answer": "wrong"}]
    questions = [{"question_type": "counterfactual", "question": "Which?", "choices": choices},
                 {"question_type": "counterfactual", "question": "Empty?", "choices": [choices[1]]}]
    rows = convert([{"video_filename": "video_00000.mp4", "questions": questions}], tmp_path)
    assert len(rows) == 1 and rows[0]["solution"] == "<answer>B</answer>"
    assert rows[0]["problem"] == "Which?\nA. first\nB. second"
