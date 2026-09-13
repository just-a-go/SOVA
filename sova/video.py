"""Shared, explicit 16-frame video preprocessing for Qwen2.5-VL."""

import av
import numpy as np
from PIL import Image
from qwen_vl_utils.vision_process import smart_resize


def read_video(path, max_pixels, frames=16):
    # Two streaming passes avoid keeping the entire decoded video in memory.
    with av.open(str(path)) as container:
        total = sum(1 for _ in container.decode(video=0))
    if total == 0:
        raise ValueError(f"Video contains no decodable frames: {path}")
    indices = np.linspace(0, total - 1, frames).round().astype(int).tolist()
    selected = {}
    needed = set(indices)
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in needed:
                image = frame.to_image().convert("RGB")
                height, width = smart_resize(image.height, image.width, min_pixels=4 * 28 * 28, max_pixels=max_pixels)
                selected[index] = np.asarray(image.resize((width, height), Image.Resampling.BICUBIC))
            if index >= indices[-1]:
                break
    return np.stack([selected[index] for index in indices])


TRAIN_PROMPT = "{question} Output the thinking process in <think> </think> and final answer (letters separated by , if multiple) in <answer> </answer> tags."
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think><answer> answer here </answer>"
)
GENERAL_PROMPT = (
    "{question} Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', "
    "'let\'s break it down', etc, or other natural language thought expressions. "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer "
    "between the <answer> and </answer> tags."
)


def messages(row, evaluation=False, general=False):
    question = row["problem"]
    if row.get("options"):
        question += "\nOptions:\n" + "\n".join(row["options"])
    prompt = GENERAL_PROMPT if general else TRAIN_PROMPT
    text = prompt.format(question=question)
    if general:
        text += " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
    content = [{"type": "video"}, {"type": "text", "text": text}]
    result = [{"role": "user", "content": content}]
    return [{"role": "system", "content": SYSTEM_PROMPT}] + result if evaluation and not general else result
