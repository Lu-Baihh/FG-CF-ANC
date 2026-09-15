"""Offline inference with the released complete ANC model checkpoint.

This script loads ``ckpt/full_model_final.tar`` and evaluates deterministic
seen and unseen acoustic-path segments from the official CCF dataset.  The
separate ``streaming/`` package remains the sample-by-sample deployment path.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import FullANCModel


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        default="data/CCF_dataset",
        help="Official dataset directory containing NOISE/, EXPECTED_NOISE/, and sh.npy.",
    )
    parser.add_argument(
        "--config", default="inference_config.yaml",
        help="Inference-only architecture and evaluation configuration.",
    )
    parser.add_argument(
        "--checkpoint", default="ckpt/full_model_final.tar",
        help="Released complete-model checkpoint.",
    )
    parser.add_argument(
        "--output-dir", default="exp/infer_full_model"
    )
    parser.add_argument(
        "--splits", nargs="+", choices=("seen", "unseen"),
        default=("seen", "unseen"),
    )
    parser.add_argument("--seen-path-indices", nargs="+", type=int,
                        default=tuple(range(8)))
    parser.add_argument("--unseen-path-indices", nargs="+", type=int,
                        default=(8, 9))
    parser.add_argument("--holdout-noise-count", type=int, default=2)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--segment-duration", type=float, default=2.0)
    parser.add_argument("--skip-seconds", type=float, default=20.0)
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', or a torch device such as 'cuda:0'.")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Optional cap per split for a quick smoke test.")
    return parser.parse_args()


def read_audio(path, start, frames):
    audio, sample_rate = sf.read(
        path, start=int(start), frames=int(frames), dtype="float32", always_2d=False
    )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < frames:
        audio = np.pad(audio, (0, frames - len(audio)))
    return audio, int(sample_rate)


def load_secondary_paths(dataset_dir, required_paths):
    paths = np.load(Path(dataset_dir) / "sh.npy")
    if paths.ndim != 2:
        raise ValueError(f"Expected sh.npy to be two-dimensional, got {paths.shape}.")
    if paths.shape[0] >= required_paths and paths.shape[0] <= paths.shape[1]:
        return paths.astype(np.float32, copy=False)
    if paths.shape[1] >= required_paths:
        return paths.T.astype(np.float32, copy=False)
    raise ValueError("sh.npy does not contain every requested acoustic path.")


def evaluation_start(path, skip_samples, frames, offset):
    total_frames = sf.info(path).frames
    maximum = total_frames - frames
    if maximum < 0:
        return 0
    start = skip_samples + int(offset)
    if start > maximum:
        start = skip_samples + (start % max(1, maximum - skip_samples + 1))
    return max(0, min(start, maximum))


def make_segment(dataset_dir, noise_names, path_index, segment_length, skip_samples, index):
    """Build the deterministic two-noise evaluation segment used by the release."""
    raw_dir = Path(dataset_dir) / "NOISE"
    expected_dir = Path(dataset_dir) / "EXPECTED_NOISE"
    first_name, second_name = noise_names[0], noise_names[-1]
    first_frames = segment_length // 2
    second_frames = segment_length - first_frames
    starts, frames = [], (first_frames, second_frames)
    references, disturbances = [], []
    for name, count in zip((first_name, second_name), frames):
        raw_path = raw_dir / f"{name}.wav"
        expected_path = expected_dir / f"{name}_scene_{path_index + 1:02d}.wav"
        if not expected_path.is_file():
            raise FileNotFoundError(f"Missing expected-noise file: {expected_path}")
        start = evaluation_start(raw_path, skip_samples, count, index * first_frames)
        reference, reference_rate = read_audio(raw_path, start, count)
        disturbance, disturbance_rate = read_audio(expected_path, start, count)
        if reference_rate != disturbance_rate:
            raise ValueError(f"Sample-rate mismatch for {name}, scene {path_index + 1}.")
        starts.append(start)
        references.append(reference)
        disturbances.append(disturbance)
    return (
        np.concatenate(references),
        np.concatenate(disturbances),
        {"scene": path_index + 1, "path_index": path_index,
         "noise_names": [first_name, second_name], "starts": starts,
         "frames": list(frames)},
    )


def load_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Inference configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but CUDA is unavailable.")
    return device


def load_full_model(config, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Complete-model checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise TypeError("Expected a complete-model archive with a 'model' state dictionary.")
    state_dict = {
        key.removeprefix("module."): value for key, value in checkpoint["model"].items()
    }
    kernels = state_dict.get("fir_kernels")
    if kernels is None or kernels.ndim != 3 or kernels.shape[1] != 1:
        raise ValueError("The complete checkpoint has an invalid Filter Experts kernel bank.")
    model = FullANCModel(
        config, num_experts=int(kernels.shape[0]), fir_taps=int(kernels.shape[-1])
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def apply_secondary_path_chunk(controller_output, secondary_path, history=None):
    """Apply one causal secondary-path chunk and preserve its state."""
    batch, _ = controller_output.shape
    path_length = secondary_path.shape[1]
    history_length = path_length - 1
    if history is None:
        history = controller_output.new_zeros(batch, history_length)
    extended = torch.cat((history, controller_output), dim=1)
    anti_noise = F.conv1d(
        extended.view(1, batch, -1),
        torch.flip(secondary_path, dims=[1]).view(batch, 1, path_length),
        groups=batch,
    ).squeeze(0)
    return anti_noise, extended[:, -history_length:].detach() if history_length else extended[:, :0]


def run_full_model(model, reference, disturbance, secondary_path, chunk_size, device):
    """Run the full checkpoint with its two-block causal feedback convention."""
    reference = torch.from_numpy(reference).to(device).unsqueeze(0)
    disturbance = torch.from_numpy(disturbance).to(device).unsqueeze(0)
    secondary_path = torch.from_numpy(secondary_path).to(device).unsqueeze(0)
    zeros = reference.new_zeros(1, chunk_size)
    reference_queue = [zeros.clone(), zeros.clone()]
    output_queue = [zeros.clone(), zeros.clone()]
    error_queue = [zeros.clone(), zeros.clone()]
    reference_history = None
    secondary_history = None
    residual_chunks = []
    with torch.inference_mode():
        for start in range(0, reference.shape[1], chunk_size):
            end = min(start + chunk_size, reference.shape[1])
            wavenet_chunk = model.compute_frozen_wavenet_chunk(reference, start, end)
            feedback_reference = reference_queue.pop(0)
            feedback_output = output_queue.pop(0)
            feedback_error = error_queue.pop(0)
            controller_output, _weights, reference_history = model(
                wavenet_chunk, reference[:, start:end], reference_history,
                feedback_reference, feedback_output, feedback_error,
            )
            anti_noise, secondary_history = apply_secondary_path_chunk(
                controller_output, secondary_path, secondary_history
            )
            residual = disturbance[:, start:end] - anti_noise
            reference_queue.append(reference[:, start:end])
            output_queue.append(controller_output)
            error_queue.append(residual)
            residual_chunks.append(residual)
    return torch.cat(residual_chunks, dim=1)[0].cpu().numpy()


def spectral_metrics(disturbance, residual, sample_rate):
    """Compute broadband and 1/3-octave-style NR/rebound release metrics."""
    epsilon = 1e-20
    n_fft = min(8192, len(disturbance))
    if n_fft < 32:
        raise ValueError("Segments must contain at least 32 samples.")
    hop = max(1, n_fft // 4)
    window = np.hanning(n_fft).astype(np.float64)

    def power(signal):
        if len(signal) <= n_fft:
            frames = [np.pad(signal, (0, max(0, n_fft - len(signal))))]
        else:
            frames = [signal[start:start + n_fft] for start in range(0, len(signal) - n_fft + 1, hop)]
        spectrum = np.fft.rfft(np.asarray(frames) * window, axis=1)
        return np.mean(np.abs(spectrum) ** 2, axis=0) / np.sum(window ** 2)

    disturbance_power, residual_power = power(disturbance), power(residual)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    centers = []
    center = 50.0
    while center <= min(16000.0, sample_rate / 2):
        centers.append(center)
        center *= 2.0 ** (1.0 / 3.0)
    changes = []
    for center in centers:
        lower, upper = center / (2.0 ** (1.0 / 6.0)), center * (2.0 ** (1.0 / 6.0))
        band = (frequencies >= lower) & (frequencies < upper)
        if np.any(band):
            changes.append((center, 10.0 * np.log10((disturbance_power[band].sum() + epsilon) /
                                                     (residual_power[band].sum() + epsilon))))
    nr = [value for center, value in changes if 50.0 <= center <= 5000.0]
    rebound = [-value for center, value in changes if 1000.0 <= center <= 16000.0]
    broadband = 10.0 * np.log10((np.sum(disturbance ** 2) + epsilon) /
                                 (np.sum(residual ** 2) + epsilon))
    third_octave_nr = float(np.mean(nr)) if nr else float("nan")
    rebound_peak = max(0.0, float(np.max(rebound))) if rebound else 0.0
    return {
        "broadband_noise_reduction_db": float(broadband),
        "noise_reduction_50hz_5khz_db": third_octave_nr,
        "rebound_peak_db": rebound_peak,
        "final_nr_minus_rebound_db": third_octave_nr - rebound_peak,
        "nmse_db": float(-broadband),
    }


def save_spectrum(path, disturbance, residual, sample_rate, title):
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.psd(disturbance, NFFT=1024, Fs=sample_rate, label="Primary noise", color="black")
    axis.psd(residual, NFFT=1024, Fs=sample_rate, label="ANC residual", color="#0066CC")
    axis.set_xscale("log")
    axis.set_xlim(left=20.0, right=min(16000.0, sample_rate / 2))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power Spectral Density (dB)")
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.5)
    axis.legend(loc="lower left")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device)
    model, checkpoint = load_full_model(config, args.checkpoint, device)
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("--max-items must be positive when supplied.")
    noise_names = sorted(path.stem for path in (dataset_dir / "NOISE").glob("*.wav"))
    if len(noise_names) <= args.holdout_noise_count:
        raise ValueError("Not enough noise files for the requested held-out split.")
    held_out = noise_names[-args.holdout_noise_count:]
    split_paths = {"seen": tuple(args.seen_path_indices), "unseen": tuple(args.unseen_path_indices)}
    required_paths = max((*split_paths["seen"], *split_paths["unseen"])) + 1
    secondary_paths = load_secondary_paths(dataset_dir, required_paths)
    segment_length = int(args.segment_duration * args.sample_rate)
    if segment_length < 8192:
        raise ValueError("--segment-duration must produce at least 8,192 samples.")
    skip_samples = int(args.skip_seconds * args.sample_rate)
    output_dir = Path(args.output_dir)
    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    metric_fields = (
        "broadband_noise_reduction_db", "noise_reduction_50hz_5khz_db",
        "rebound_peak_db", "final_nr_minus_rebound_db", "nmse_db",
    )
    chunk_size = int(config["model"]["gating_block_size"])
    print(f"Using complete checkpoint on {device}: {Path(args.checkpoint).resolve()}")
    for split in args.splits:
        paths = split_paths[split]
        if args.max_items is not None:
            paths = paths[:args.max_items]
        for index, path_index in enumerate(paths):
            reference, disturbance, meta = make_segment(
                dataset_dir, held_out, path_index, segment_length, skip_samples, index
            )
            residual = run_full_model(
                model, reference, disturbance, secondary_paths[path_index], chunk_size, device
            )
            metrics = spectral_metrics(disturbance, residual, args.sample_rate)
            rows.append({
                "split": split, "dataset_index": index, **meta,
                "noise_names": " | ".join(meta["noise_names"]),
                "starts": json.dumps(meta["starts"]), "frames": json.dumps(meta["frames"]),
                **metrics,
            })
            save_spectrum(
                spectra_dir / f"{split}_scene_{meta['scene']:02d}_segment_{index:02d}_spectrum.png",
                disturbance, residual, args.sample_rate, f"{split.title()} | Scene {meta['scene']:02d}",
            )
            print(f"[{split} {index + 1}/{len(paths)}] scene={meta['scene']:02d} "
                  f"NR50-5k={metrics['noise_reduction_50hz_5khz_db']:.2f} dB")
    fields = ("split", "dataset_index", "scene", "path_index", "noise_names", "starts", "frames", *metric_fields)
    write_csv(output_dir / "segment_metrics.csv", rows, fields)
    averages = []
    for split in (*args.splits, "all"):
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        if selected:
            averages.append({"split": split, "num_segments": len(selected),
                             **{field: float(np.mean([row[field] for row in selected])) for field in metric_fields}})
    write_csv(output_dir / "average_metrics.csv", averages, ("split", "num_segments", *metric_fields))
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as file:
        json.dump({"dataset_dir": str(dataset_dir), "sample_rate": args.sample_rate,
                   "segment_duration": args.segment_duration, "splits": list(args.splits),
                   "segments_processed": len(rows), "controller": str(Path(args.checkpoint)),
                   "checkpoint_format": checkpoint.get("format"),
                   "checkpoint_epoch": checkpoint.get("epoch")}, file, indent=2)
    print(f"Saved {len(rows)} PSD figures and CSV metrics to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
