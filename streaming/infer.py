"""Public sample-by-sample streaming ANC inference interface."""

from __future__ import annotations

from pathlib import Path

import torch

from .code import StreamingHybridANC


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "model" / "full_model_streaming.pth"


class StreamingANCInference:
    """Stateful, strictly causal controller used by a real-time audio loop."""

    sample_rate = 48_000
    requires_error = True

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        if self.device.type == "cpu" and torch.get_num_threads() > 1:
            torch.set_num_threads(1)
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"Missing streaming model: {CHECKPOINT_PATH}")
        self.model = StreamingHybridANC(CHECKPOINT_PATH, self.device)

    def reset(self) -> None:
        """Clear all controller state before a new audio stream."""
        self.model.reset_streaming_state()

    def process_sample(
        self, reference_sample: float, previous_error_sample: float
    ) -> float:
        """Return the controller output for one reference/error sample pair."""
        return self.model.process_sample(reference_sample, previous_error_sample)

    def get_complexity(self) -> dict[str, int]:
        return {
            "parameter_count": self.model.parameter_count,
            "steady_state_macs_per_sample": self.model.steady_state_macs_per_sample,
            "startup_macs": 0,
            "peak_macs_in_one_sample_event": self.model.peak_macs_in_one_sample_event,
        }


def create_model(device: str = "cpu") -> StreamingANCInference:
    """Create the released streaming controller."""
    return StreamingANCInference(device=device)
