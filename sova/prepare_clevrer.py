"""Convert official CLEVRER question files to the paper's counterfactual QA format."""

import argparse
import json
from pathlib import Path


def convert(scenes, video_root):
    videos = {}
    for path in Path(video_root).rglob("*.mp4"):
        if path.name in videos:
            raise ValueError(f"Duplicate video filename: {path.name}")
        videos[path.name] = str(path.resolve())
    rows = []
    for scene in scenes:
        for question in scene["questions"]:
            if question["question_type"] != "counterfactual":
                continue
            choices = sorted(question["choices"], key=lambda item: item["choice_id"])
            if not 1 <= len(choices) <= 26 or any(choice.get("answer") not in {"correct", "wrong"} for choice in choices):
                raise ValueError("Use labeled training/validation questions with at most 26 choices")
            answers = [chr(65 + index) for index, choice in enumerate(choices) if choice["answer"] == "correct"]
            if not answers:
                continue  # The paper assumes nonempty ground-truth answer sets.
            video = videos.get(scene["video_filename"])
            if video is None:
                raise FileNotFoundError(scene["video_filename"])
            options = [f"{chr(65 + index)}. {choice['choice']}" for index, choice in enumerate(choices)]
            rows.append({"video": video, "problem": question["question"] + "\n" + "\n".join(options),
                         "solution": "<answer>" + ",".join(answers) + "</answer>"})
    if not rows:
        raise ValueError("No labeled counterfactual questions with nonempty answers were found")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, help="Official questions/train.json or val.json")
    parser.add_argument("--videos", required=True, help="Extracted video directory (searched recursively)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Output already exists; choose a new filename")
    with open(args.questions, encoding="utf-8") as handle:
        rows = convert(json.load(handle), args.videos)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Prepared {len(rows)} counterfactual questions in {output}")


if __name__ == "__main__":
    main()
