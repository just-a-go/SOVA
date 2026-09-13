"""Small random Qwen2.5-VL: real video preprocessing, rollout, loss, and optimizer."""

import av
import json
import sys
import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLConfig,
                          Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, TrainingArguments)

from sova.trainer import SOVATrainer
from sova.video import read_video


def tiny_video(path):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width = stream.height = 56
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((56, 56, 3), index * 25, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def tiny_processor():
    special = ["<unk>", "<pad>", "<eos>", "<|vision_start|>", "<|vision_end|>", "<|video_pad|>", "<|image_pad|>"]
    vocab = {token: index for index, token in enumerate(special + ["A", "B", "C", "think", "answer"])}
    backend = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer = Qwen2TokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>",
                                   additional_special_tokens=special[3:])
    tokenizer.padding_side = "left"
    tokenizer.chat_template = "<|vision_start|><|video_pad|><|vision_end|> A"
    return Qwen2_5_VLProcessor(Qwen2VLImageProcessor(min_pixels=3136, max_pixels=3136), tokenizer,
                             chat_template=tokenizer.chat_template)


def test_video_and_actual_training_step(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    path = tmp_path / "synthetic.mp4"
    tiny_video(path)
    video = read_video(path, 3136)
    assert video.shape == (16, 56, 56, 3)
    assert video[0].mean() < video[-1].mean()
    processor = tiny_processor()
    config = Qwen2_5_VLConfig(
        vocab_size=len(processor.tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=256, rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_start_token_id=3, vision_end_token_id=4, video_token_id=5, image_token_id=6,
        eos_token_id=2, pad_token_id=1,
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                       "out_hidden_size": 32, "num_heads": 2, "fullatt_block_indexes": [0]},
    )
    model = Qwen2_5_VLForConditionalGeneration(config)
    # Keep the random language model from sampling multimodal control tokens.
    arguments = TrainingArguments(output_dir=str(tmp_path / "run"), use_cpu=True,
                                  per_device_train_batch_size=1, max_steps=1,
                                  learning_rate=1e-3, report_to="none", save_strategy="no",
                                  remove_unused_columns=False, disable_tqdm=True)
    trainer = SOVATrainer(model=model, args=arguments, processing_class=processor,
                          train_dataset=[{"video": str(path), "problem": "Select A", "solution": "Z"}],
                          data_collator=lambda rows: rows, group_size=2, max_length=64)
    trainer.generation.max_new_tokens = 4
    trainer.generation.suppress_tokens = [3, 4, 5, 6]
    before = model.lm_head.weight.detach().clone()
    result = trainer.train()
    assert result.global_step == 1 and np.isfinite(result.training_loss)
    assert not torch.equal(before, model.lm_head.weight.detach())

    # Exercise the real evaluator and result serialization using the same tiny model.
    from sova import evaluate
    data_path, output_path = tmp_path / "data.json", tmp_path / "results.json"
    data_path.write_text(json.dumps([{"video": str(path), "problem": "Select A", "solution": "A"}]))
    monkeypatch.setattr(evaluate.AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(evaluate.AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(evaluate.Qwen2_5_VLForConditionalGeneration, "from_pretrained", lambda *a, **kw: model)
    generate = model.generate
    def short_generate(**kwargs):
        kwargs.update(max_new_tokens=4, suppress_tokens=[3, 4, 5, 6])
        return generate(**kwargs)
    monkeypatch.setattr(model, "generate", short_generate)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--model", "tiny", "--data", str(data_path),
                                      "--output", str(output_path), "--benchmark", "clevrer"])
    evaluate.main()
    saved = json.loads(output_path.read_text())
    assert saved["metrics"]["samples"] == 1
    assert 0 <= saved["metrics"]["strict_accuracy"] <= 100
    assert isinstance(saved["results"][0]["response"], str)
