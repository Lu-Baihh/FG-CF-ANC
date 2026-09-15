# Feedback-Guided DNN-Based Controller Fusion for Robust Fixed-Parameter Active Noise Control

This repository is the official inference release for
*Feedback-Guided DNN-Based Controller Fusion for Robust Fixed-Parameter Active
Noise Control*.

Fixed-parameter ANC controllers avoid the instability and convergence issues of
online adaptation, but can lose effectiveness when noise characteristics or
acoustic paths change. The proposed feedback-guided controller fusion framework
combines a causal **Feedforward WaveNet** branch with a feedback-guided
mixture-of-experts branch. A **Gating Network** uses the reference signal,
controller output, and previous error signal to fuse multiple pre-trained
**Filter Experts** according to the current acoustic condition.

The system is fully causal and supports sample-by-sample streaming inference.
Its mixed 2,048-tap FIR controller is built progressively across sampling
points, avoiding a large block-boundary computational peak. This repository
includes the released final-model archive, an offline inference/evaluation
script, result figures, and the minimal streaming deployment implementation.

Evaluation audio and generated inference output are excluded. See
[DATASET.md](DATASET.md) for the data layout and [ckpt/README.md](ckpt/README.md)
for the released checkpoint layout.

> **Dataset availability.** The reported results use the official CCF competition
> dataset. It is not redistributed with this repository. Please wait for the
> official public dataset release, and obtain and use the data only under the
> competition organizer's published terms and rules. Official competition
> resources and announcements are available at
> [CCF2026ANC/CCF_DEEPANC_2026](https://github.com/CCF2026ANC/CCF_DEEPANC_2026).

## Repository layout

```text
ckpt/           Released complete-model archive
model/          Minimal architecture needed to load the complete checkpoint
inference_config.yaml  Inference-only model and evaluation configuration
infer.py        Offline full-model evaluation, PSD figures, and metric CSV export
streaming/      Released causal deployment model and public entry point
assets/figures/ Published frequency-domain result figures
```

## Setup

Use Python 3.10+:

```bash
python -m pip install -r requirements.txt
```

After the official release, place the obtained dataset at
`data/CCF_dataset/`, as described in `DATASET.md`. Run all commands from the
repository root.

## Offline inference

Run the released complete model over the deterministic seen/unseen acoustic-path
segments:

```bash
python infer.py \
  --dataset-dir data/CCF_dataset \
  --output-dir exp/infer_full_model
```

This produces one PSD figure per segment in `spectra/`, detailed segment
metrics in `segment_metrics.csv`, and Seen/Unseen/all averages in
`average_metrics.csv`.

`infer.py` directly loads `ckpt/full_model_final.tar`. The inference-only
`model/` package and `inference_config.yaml` supply the architecture required
to import that complete checkpoint; both are used only for inference.

## Frequency-domain results

The released checkpoints and the results reported below use the **`path9,10`**
split: acoustic paths 9 and 10 are held out for evaluation. The 20 PSD figures
in this primary result set cover the eight seen paths and the two held-out
paths. Each figure compares the primary noise with the ANC residuals; the
panels separately compare the baseline with the Feedforward WaveNet branch, or
the WaveNet branch with the feedback-guided Filter Experts mixture.

The labels “unseen path 1” and “unseen path 2” inside the `path9,10` figures
refer to the first and second held-out evaluation paths (i.e., paths 9 and 10,
respectively).

<details open>
<summary><strong>Seen paths: Baseline and Feedforward-only</strong></summary>

<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_1_freq.png" width="49%" alt="Seen path 1: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_2_freq.png" width="49%" alt="Seen path 2: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_3_freq.png" width="49%" alt="Seen path 3: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_4_freq.png" width="49%" alt="Seen path 4: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_5_freq.png" width="49%" alt="Seen path 5: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_6_freq.png" width="49%" alt="Seen path 6: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_7_freq.png" width="49%" alt="Seen path 7: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_seen_paths_baseline_feedforward_path_8_freq.png" width="49%" alt="Seen path 8: baseline and Feedforward-only">

</details>

<details open>
<summary><strong>Seen paths: WaveNet and Filter Experts mixture</strong></summary>

<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_1_freq.png" width="49%" alt="Seen path 1: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_2_freq.png" width="49%" alt="Seen path 2: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_3_freq.png" width="49%" alt="Seen path 3: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_4_freq.png" width="49%" alt="Seen path 4: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_5_freq.png" width="49%" alt="Seen path 5: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_6_freq.png" width="49%" alt="Seen path 6: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_7_freq.png" width="49%" alt="Seen path 7: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_seen_paths_wavenet_moe_path_8_freq.png" width="49%" alt="Seen path 8: WaveNet and Filter Experts mixture">

</details>

<details open>
<summary><strong>Unseen paths: Baseline and Feedforward-only</strong></summary>

<img src="assets/figures/path9,10/combined_unseen_paths_baseline_feedforward_path_1_freq.png" width="49%" alt="Unseen path 9: baseline and Feedforward-only">
<img src="assets/figures/path9,10/combined_unseen_paths_baseline_feedforward_path_2_freq.png" width="49%" alt="Unseen path 10: baseline and Feedforward-only">

</details>

<details open>
<summary><strong>Unseen paths: WaveNet and Filter Experts mixture</strong></summary>

<img src="assets/figures/path9,10/combined_unseen_paths_wavenet_moe_path_1_freq.png" width="49%" alt="Unseen path 9: WaveNet and Filter Experts mixture">
<img src="assets/figures/path9,10/combined_unseen_paths_wavenet_moe_path_2_freq.png" width="49%" alt="Unseen path 10: WaveNet and Filter Experts mixture">

</details>

### Additional data-split results

For completeness, the following eight figures show results from two additional
held-out path splits: **`path3,6`** and **`path1,10`**. These are supplementary
evaluation results only; the released checkpoints and the model-related files
in this repository correspond to the primary **`path9,10`** split above.

<details open>
<summary><strong>Held-out paths 3 and 6</strong></summary>

<img src="assets/figures/path3,6/combined_unseen_paths_baseline_feedforward_path_1_freq.png" width="49%" alt="Held-out path 3: baseline and Feedforward-only">
<img src="assets/figures/path3,6/combined_unseen_paths_baseline_feedforward_path_2_freq.png" width="49%" alt="Held-out path 6: baseline and Feedforward-only">
<img src="assets/figures/path3,6/combined_unseen_paths_wavenet_moe_path_1_freq.png" width="49%" alt="Held-out path 3: WaveNet and Filter Experts mixture">
<img src="assets/figures/path3,6/combined_unseen_paths_wavenet_moe_path_2_freq.png" width="49%" alt="Held-out path 6: WaveNet and Filter Experts mixture">

</details>

<details open>
<summary><strong>Held-out paths 1 and 10</strong></summary>

<img src="assets/figures/path1,10/combined_unseen_paths_baseline_feedforward_path_1_freq.png" width="49%" alt="Held-out path 1: baseline and Feedforward-only">
<img src="assets/figures/path1,10/combined_unseen_paths_baseline_feedforward_path_2_freq.png" width="49%" alt="Held-out path 10: baseline and Feedforward-only">
<img src="assets/figures/path1,10/combined_unseen_paths_wavenet_moe_path_1_freq.png" width="49%" alt="Held-out path 1: WaveNet and Filter Experts mixture">
<img src="assets/figures/path1,10/combined_unseen_paths_wavenet_moe_path_2_freq.png" width="49%" alt="Held-out path 10: WaveNet and Filter Experts mixture">

</details>

## Streaming deployment

`streaming/model/full_model_streaming.pth` is the already-exported,
inference-only streaming weight file. The public entry point is
`streaming.infer:create_model`:

```python
from streaming.infer import create_model

model = create_model(device="cpu")
model.reset()
controller_output = model.process_sample(reference, previous_error)
```

It is strictly causal and processes one sample at a time. The mixed 2,048-tap
FIR kernel is built one coefficient per sample, avoiding a block-boundary peak
in computation.

The streaming checkpoint is intentionally a separate deployment export. It is
used through `streaming.infer:create_model`, whereas `infer.py` uses the
complete checkpoint in `ckpt/` for offline evaluation.

## License

This project is released under the [MIT License](LICENSE).
