<div align="center">

# 🎬 SOVA: Strict Outcome-Conditioned Virtual Advantages for Video Reasoning

**Recovering correctness signals at outcome extremes — without extra rollouts.**

🤗 [Qwen2.5-VL Backbone](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) · 📑 [TW-GRPO Paper](https://arxiv.org/abs/2505.24718) · 🧮 [SOVA Implementation](sova/core.py)

</div>

🚀 Official implementation of **SOVA: Strict Outcome-Conditioned Virtual Advantages for Video Reasoning**.

Group-relative reinforcement learning relies on differences between sampled responses. When every response is correct, its learning signal disappears. When every response is incorrect, formatting can become the only source of reward contrast. **SOVA** recovers a correctness-oriented group signal by adding a fixed virtual reward to the normalization statistics and interpolating the resulting advantages with the original estimates.

Built on **TW-GRPO**, SOVA retains token importance weighting and partial-credit rewards. Virtual anchors contribute **no response tokens, no extra policy rollouts, and no inference-time computation**.

## 🔥 Innovation

**Strict outcomes guide advantage calibration.** SOVA uses accuracy and format rewards to select two complementary branches:

| Branch | When it activates | Virtual reward | Effect |
| :-- | :-- | :--: | :-- |
| **Positive** | Every response is fully correct and well formatted | 1.5 | Restores positive advantages |
| **Negative** | Every response receives zero accuracy credit | 2.0 | Produces a negative group mean while retaining format ordering |

All other groups keep their original advantages exactly. In negative groups with mixed formatting, individual advantages can remain positive; the guarantee concerns the **group mean**.

## ✨ Highlights

- 🎯 **Outcome-conditioned correction.** Component-level gates distinguish strict successes, complete accuracy failures, and partially correct groups.
- 🧮 **Simple, explicit statistics.** One scalar anchor, sample standard deviation, and a small residual interpolation implement the correction.
- 🔄 **A matched TW-GRPO control.** Set both interpolation coefficients to zero; the data, rewards, token weights, and update pipeline stay identical.
- 🪶 **Minimal source release.** Training, evaluation, video preprocessing, and focused tests. Download datasets and model weights separately.

## 📊 Results

The formal experiments were run on H800 servers. The SOVA manuscript reports the following results with Qwen2.5-VL-7B, 1,000 CLEVRER training queries, 500 updates, and group size 8. Scores are accuracy (%); Video-MME excludes subtitles.

| Method | CLEVRER | NExT-GQA | MMVU | MVBench | TempCompass | Video-MME |
| :-- | --: | --: | --: | --: | --: | --: |
| TW-GRPO control | 50.4 | 76.1 | 65.8 | 63.3 | **73.3** | 55.1 |
| **SOVA** | **51.9** | **76.7** | **65.9** | **64.4** | **73.3** | **56.9** |

This compact implementation targets the main **Qwen2.5-VL-7B** protocol and the **Qwen2.5-VL-3B** backbone. The paper's InternVL3 implementation and unrelated baseline methods are outside this release. The table reports manuscript results, not a rerun of this source release.

## 🛠️ Setup

Use **Linux, Python 3.10/3.11, and two NVIDIA H800 80 GB GPUs** for the reference training run. Evaluation uses one GPU; the reference batch size is 16.

```bash
git clone https://github.com/just-a-go/SOVA.git
cd SOVA
conda create -n sova python=3.10 -y
conda activate sova
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[train,test]"
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Set `CUDA_HOME` to your CUDA toolkit installation if it is not already configured. CUDA 12.4 and a C++ build toolchain are needed to build the training extensions.

### 📥 Model backbone

```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir models/Qwen2.5-VL-7B-Instruct
```

### 🎥 Datasets

Download the videos and annotations from their original sources:

| Dataset | Source | Subset |
| :-- | :-- | :-- |
| CLEVRER | [Official website](https://clevrer.csail.mit.edu/) | Counterfactual |
| NExT-GQA | [Official repository](https://github.com/doc-doc/NExT-GQA) | Multiple choice |
| MMVU | [Hugging Face](https://huggingface.co/datasets/yale-nlp/MMVU) | Multiple choice |
| MVBench | [Hugging Face](https://huggingface.co/datasets/OpenGVLab/MVBench) | Multiple choice |
| TempCompass | [Hugging Face](https://huggingface.co/datasets/lmms-lab/TempCompass) | Multiple choice |
| Video-MME | [Hugging Face](https://huggingface.co/datasets/lmms-lab/Video-MME) | Without subtitles |

Each input file is a **JSON array**. Use absolute video paths. For CLEVRER, place the question and labeled choices in `problem`; comma-separated `solution` letters represent all correct choices. For the other benchmarks, provide labeled choices in `options`.

```json
[
  {
    "video": "/absolute/path/to/video.mp4",
    "problem": "Your question.\nA. First choice\nB. Second choice",
    "solution": "<answer>A,B</answer>"
  }
]
```

The example illustrates the schema; it is not a benchmark sample. Supply at least 1,000 training queries. The loader selects one fixed 1,000-query subset with seed 42; `--seed` controls the training shuffle and rollout sampling. Keep the annotation order unchanged when comparing runs.

For CLEVRER, convert the official **Questions and Answers** file after extracting the videos:

```bash
python -m sova.prepare_clevrer \
  --questions /absolute/path/to/questions/train.json \
  --videos /absolute/path/to/train_videos \
  --output data/clevrer_train.json
```

Repeat for validation. The converter retains counterfactual questions with nonempty correct-answer sets, as assumed in the paper, and resolves nested video directories. For the other benchmarks, adapt their annotations to the schema above with `options` containing labeled choices.

## 🏃 Training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 -m sova.train \
  --model models/Qwen2.5-VL-7B-Instruct \
  --data /absolute/path/to/clevrer_train.json \
  --output outputs/sova
```

The defaults implement the reference configuration:

| Setting | Value |
| :-- | :-- |
| Training queries / updates | 1,000 / 500 |
| Query batch / processes | 1 per device / 2 |
| Group size | 8 responses per query |
| SOVA coefficients | `lambda_positive=0.0625`, `lambda_negative=0.03125` |
| Virtual rewards / normalization epsilon | 1.5 and 2.0 / 0.0001 |
| TW coefficient / maximum token weight | 0.7 / 1.7 |
| KL coefficient / ratio clipping | 0 / [0.8, 1.2] |
| Learning rate / gradient clipping | 0.000001 / 20 |
| Prompt / completion limit | 4,096 / 4,096 tokens |
| Training frames / pixel budget per frame | 16 uniformly sampled / 128 × 28 × 28 |
| Precision / optimizer sharding | BF16 / ZeRO-3 with CPU offloading |
| Vision encoder | Frozen |

One fresh response group is used once per update. Token weighting retains the original TW-GRPO rule and its gradient path; SOVA changes only the response advantages. Checkpointed token chunks avoid retaining a second full-vocabulary log-probability tensor. No reward-model, reference-policy, or rollout-reuse branch is needed for the paper's `beta=0` setting.

### ⚙️ Controlled comparisons

Append these options to the training command and choose a separate output directory:

| Configuration | Options |
| :-- | :-- |
| TW-GRPO control | `--lambda-positive 0 --lambda-negative 0` |
| Positive-only SOVA | `--lambda-negative 0` |
| Negative-only SOVA | `--lambda-positive 0` |
| Group-size study | `--group-size 2`, `4`, `8`, or `16` |
| Paired seeds | `--seed 11`, `22`, `33`, `44`, or `55` |

Model and processor files are saved together in the output directory. Existing nonempty output directories are rejected to keep runs separate.

## 📈 Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m sova.evaluate \
  --model outputs/sova \
  --data /absolute/path/to/clevrer_val.json \
  --benchmark clevrer \
  --output outputs/sova/clevrer_results.json
```

Use `nextgqa`, `mmvu`, `mvbench`, `tempcompass`, or `videomme` with the corresponding annotation file. The evaluator uses **16 frames, a 256 × 28 × 28 pixel budget per frame, temperature 0.01, top-p 0.001, and batch size 16**. It reports strict answer-set accuracy, partial credit, and format compliance, and saves each response. A lower `--batch-size` can reduce memory use but differs from the reported evaluation protocol.

## ✅ Check the implementation

```bash
python -m pytest tests -q
```

The tests cover strict gates, sample statistics, mixed-format failures, exact bypass, group-size changes, reward parsing, EOS masks, data conversion, and TW weight gradients. A tiny randomly initialized Qwen2.5-VL model also reads a synthetic video, generates responses, completes an optimizer update, and runs the evaluator on CPU. This test downloads no dataset or pretrained checkpoint; it does not replace a full H800 benchmark reproduction.

## 📁 Code

| File | Purpose |
| :-- | :-- |
| [`sova/core.py`](sova/core.py) | Rewards, SOVA advantages, TW weights, and policy loss |
| [`sova/video.py`](sova/video.py) | Uniform frame sampling and prompts |
| [`sova/trainer.py`](sova/trainer.py) | Group rollouts and policy updates |
| [`sova/train.py`](sova/train.py) | Reference training entry point |
| [`sova/evaluate.py`](sova/evaluate.py) | Six-benchmark multiple-choice evaluation |
| [`sova/prepare_clevrer.py`](sova/prepare_clevrer.py) | Official CLEVRER annotation conversion |

## 🙏 Acknowledgements

We thank [TW-GRPO](https://github.com/longmalongma/TW-GRPO), [Open-R1-Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video), [Video-R1](https://github.com/tulerfeng/Video-R1), [VideoChat-R1](https://github.com/OpenGVLab/VideoChat-R1), [TRL](https://github.com/huggingface/trl), and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for their open-source contributions.

The SOVA paper link and citation will be added when its public manuscript is available. The preceding TW-GRPO work is [Reinforcing Video Reasoning with Focused Thinking](https://arxiv.org/abs/2505.24718).

Released under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution.
