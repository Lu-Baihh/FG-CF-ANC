"""Strictly causal, peak-MAC-flattened streaming ANC inference.

The module names match the exported model weight. The optimization follows the
T036 Task2 implementation: a newly selected 2,048-tap mixed FIR kernel is built
one coefficient per sample instead of all coefficients at a block boundary.
"""

from collections import OrderedDict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_EXPERTS = 8
FIR_TAPS = 2048
CLASSIFIER_CHANNELS = (6, 12, 12, 9)
CLASSIFIER_HEAD_CHANNELS = 8
CLASSIFIER_BLOCK_SIZE = 2048
WAVENET_DILATIONS = tuple(2**index for index in range(9)) * 2

# MACs count multiplications/accumulations only. Bias, activation, normalization,
# softmax, memory copies, and indexing are not counted, matching T036's method.
WAVENET_MACS = 10_080
FIR_FILTER_MACS = 2_048
CLASSIFIER_CONV_MACS = 1_026
CLASSIFIER_HEAD_MACS = 160
CLASSIFIER_MACS = CLASSIFIER_CONV_MACS + CLASSIFIER_HEAD_MACS
KERNEL_BUILD_MACS_PER_SAMPLE = NUM_EXPERTS
ORDINARY_STREAMING_PEAK_MACS = (
    WAVENET_MACS + FIR_FILTER_MACS + CLASSIFIER_MACS
    + NUM_EXPERTS * FIR_TAPS
)
OPTIMIZED_STREAMING_PEAK_MACS = (
    WAVENET_MACS + FIR_FILTER_MACS + CLASSIFIER_MACS
    + KERNEL_BUILD_MACS_PER_SAMPLE
)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not payload or not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload.items()
    ):
        raise RuntimeError("The inference checkpoint is not a state_dict.")
    return payload


class CausalConv1d(nn.Module):
    _cache: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.left_padding = int(dilation) * (int(kernel_size) - 1)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            dilation=int(dilation),
            bias=bool(bias),
        )
        self._cache = None

    @torch.jit.export
    def reset_streaming_state(self) -> None:
        self._cache = None

    @torch.jit.export
    def forward_sample(self, signal: torch.Tensor) -> torch.Tensor:
        if self.left_padding == 0:
            return F.conv1d(signal, self.conv.weight, self.conv.bias)
        cache = self._cache
        if cache is None:
            cache = signal.new_zeros(
                (signal.shape[0], signal.shape[1], self.left_padding)
            )
        extended = torch.cat((cache, signal), dim=-1)
        self._cache = extended[:, :, 1:].detach()
        return F.conv1d(
            extended,
            self.conv.weight,
            self.conv.bias,
            dilation=self.conv.dilation,
        )


class DilatedResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(8, 16, 3, dilation=dilation, bias=False)
        self.conv2 = CausalConv1d(8, 16, 1, bias=False)

    @torch.jit.export
    def reset_streaming_state(self) -> None:
        self.conv1.reset_streaming_state()
        self.conv2.reset_streaming_state()

    @torch.jit.export
    def forward_sample(
        self, signal: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gated = self.conv1.forward_sample(signal)
        activated = torch.tanh(gated[:, :8]) * torch.sigmoid(gated[:, 8:])
        residual_and_skip = self.conv2.forward_sample(activated)
        return signal + residual_and_skip[:, :8], residual_and_skip[:, 8:]


class StreamingWaveNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dilated_layers = nn.ModuleList(
            DilatedResidualBlock(dilation) for dilation in WAVENET_DILATIONS
        )
        self.conv0 = CausalConv1d(1, 16, 3, bias=False)
        self.conv1 = CausalConv1d(16, 8, 3, bias=False)
        self.conv2 = CausalConv1d(8, 16, 3, bias=False)
        self.conv3 = CausalConv1d(16, 1, 3, bias=False)

    @torch.jit.export
    def reset_streaming_state(self) -> None:
        self.conv0.reset_streaming_state()
        self.conv1.reset_streaming_state()
        self.conv2.reset_streaming_state()
        self.conv3.reset_streaming_state()
        for layer in self.dilated_layers:
            layer.reset_streaming_state()

    @torch.jit.export
    def forward_sample(self, signal: torch.Tensor) -> torch.Tensor:
        output = self.conv1.forward_sample(self.conv0.forward_sample(signal))
        skip_sum: Optional[torch.Tensor] = None
        for layer in self.dilated_layers:
            output, skip = layer.forward_sample(output)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        if skip_sum is None:
            raise RuntimeError("WaveNet has no residual layers.")
        output = self.conv2.forward_sample(torch.tanh(skip_sum))
        return self.conv3.forward_sample(torch.tanh(output))


class PerTimeChannelNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.norm(signal.transpose(1, 2)).transpose(1, 2)


class CausalClassifierStage(nn.Module):
    _cache: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = int(dilation) * 2
        self.conv = nn.Conv1d(
            int(in_channels), int(out_channels), 3, dilation=int(dilation)
        )
        self.norm = PerTimeChannelNorm(int(out_channels))
        self.activation = nn.SiLU()
        self._cache = None

    @torch.jit.export
    def reset_streaming_state(self) -> None:
        self._cache = None

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(F.pad(signal, (self.left_padding, 0)))))

    @torch.jit.export
    def forward_sample(self, signal: torch.Tensor) -> torch.Tensor:
        cache = self._cache
        if cache is None:
            cache = signal.new_zeros(
                (signal.shape[0], signal.shape[1], self.left_padding)
            )
        extended = torch.cat((cache, signal), dim=-1)
        self._cache = extended[:, :, 1:].detach()
        output = F.conv1d(
            extended,
            self.conv.weight,
            self.conv.bias,
            dilation=self.conv.dilation,
        )
        return self.activation(self.norm(output))


class StreamingGatingNetwork(nn.Module):
    _pooled_sum: Optional[torch.Tensor]
    _square_sum: Optional[torch.Tensor]

    def __init__(self) -> None:
        super().__init__()
        stage_channels = (3,) + CLASSIFIER_CHANNELS
        dilations = (1, 3, 9, 27)
        self.stages = nn.ModuleList(
            CausalClassifierStage(
                stage_channels[index], stage_channels[index + 1], dilations[index]
            )
            for index in range(4)
        )
        self.head = nn.Sequential(
            nn.Linear(CLASSIFIER_CHANNELS[-1] + 3, CLASSIFIER_HEAD_CHANNELS),
            nn.SiLU(),
            nn.Linear(CLASSIFIER_HEAD_CHANNELS, NUM_EXPERTS),
        )
        self._pooled_sum = None
        self._square_sum = None
        self._sample_count = 0
        self.streaming_block_size = CLASSIFIER_BLOCK_SIZE

    @torch.jit.export
    def reset_streaming_state(self) -> None:
        for stage in self.stages:
            stage.reset_streaming_state()
        self._pooled_sum = None
        self._square_sum = None
        self._sample_count = 0

    def zero_block_weights(self, device: torch.device) -> torch.Tensor:
        feedback = torch.zeros(1, 3, CLASSIFIER_BLOCK_SIZE, device=device)
        encoded = feedback
        for stage in self.stages:
            encoded = stage(encoded)
        pooled = encoded.mean(dim=-1)
        log_rms = torch.log(
            torch.sqrt(feedback.square().mean(dim=-1) + 1e-8) + 1e-8
        ).clamp(min=-12.0, max=12.0)
        return torch.softmax(self.head(torch.cat((pooled, log_rms), dim=1)), dim=1)[0]

    @torch.jit.export
    def forward_sample(
        self,
        reference: torch.Tensor,
        controller_output: torch.Tensor,
        previous_error: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        feedback = torch.cat((reference, controller_output, previous_error), dim=1)
        encoded = feedback
        for stage in self.stages:
            encoded = stage.forward_sample(encoded)
        encoded_now = encoded[:, :, 0]
        feedback_now = feedback[:, :, 0]

        pooled_sum = self._pooled_sum
        square_sum = self._square_sum
        if pooled_sum is None:
            pooled_sum = torch.zeros_like(encoded_now)
            square_sum = torch.zeros_like(feedback_now)
        if square_sum is None:
            raise RuntimeError("Classifier state was not initialized.")
        pooled_sum = pooled_sum + encoded_now
        square_sum = square_sum + feedback_now.square()
        self._pooled_sum = pooled_sum
        self._square_sum = square_sum
        self._sample_count += 1

        count = float(self._sample_count)
        log_rms = torch.log(
            torch.sqrt(square_sum / count + 1e-8) + 1e-8
        ).clamp(min=-12.0, max=12.0)
        logits = self.head(torch.cat((pooled_sum / count, log_rms), dim=1))
        weights = torch.softmax(logits, dim=1)
        if self._sample_count != self.streaming_block_size:
            return None

        result = weights[0].detach()
        for stage in self.stages:
            stage.reset_streaming_state()
        self._pooled_sum = None
        self._square_sum = None
        self._sample_count = 0
        return result


class StreamingHybridANC(nn.Module):
    """WaveNet + feedback-selected FIR bank with flattened kernel construction."""

    parameter_count = 27_783
    steady_state_macs_per_sample = OPTIMIZED_STREAMING_PEAK_MACS
    peak_macs_in_one_sample_event = OPTIMIZED_STREAMING_PEAK_MACS

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.wavenet = StreamingWaveNet()
        self.classifier = StreamingGatingNetwork()
        self.fir_kernels = nn.Parameter(
            torch.zeros(NUM_EXPERTS, 1, FIR_TAPS), requires_grad=False
        )

        state = OrderedDict(_load_checkpoint(checkpoint_path, device).items())
        self.load_state_dict(state, strict=True)
        self.to(device)
        self.requires_grad_(False)
        self.eval()

        with torch.no_grad():
            startup_weights = self.classifier.zero_block_weights(device)
            self._startup_kernel = torch.mv(
                self.fir_kernels[:, 0, :].transpose(0, 1), startup_weights
            ).detach()
        self.wavenet = torch.jit.script(self.wavenet)
        self.classifier = torch.jit.script(self.classifier)
        self._sample_tensor = torch.zeros(1, 1, 1, device=device)
        self._error_tensor = torch.zeros(1, 1, 1, device=device)
        self._fir_history = torch.zeros(FIR_TAPS, device=device)
        self._active_kernel = self._startup_kernel.clone()
        self._building_kernel = torch.empty_like(self._active_kernel)
        self._build_weights: Optional[torch.Tensor] = None
        self._build_index = 0
        self._history_index = 0
        self.reset_streaming_state()

    def reset_streaming_state(self) -> None:
        self.wavenet.reset_streaming_state()
        self.classifier.reset_streaming_state()
        self._fir_history.zero_()
        self._active_kernel.copy_(self._startup_kernel)
        self._building_kernel.zero_()
        self._build_weights = None
        self._build_index = 0
        self._history_index = 0

    def _fir_sample(self, reference_value: torch.Tensor) -> torch.Tensor:
        write_index = self._history_index
        self._fir_history[write_index] = reference_value
        self._history_index = (write_index + 1) % FIR_TAPS
        split = self._history_index
        first_count = FIR_TAPS - split
        result = torch.dot(
            self._active_kernel[:first_count], self._fir_history[split:]
        )
        if split:
            result = result + torch.dot(
                self._active_kernel[first_count:], self._fir_history[:split]
            )
        return result.reshape(1, 1, 1)

    def _advance_kernel_builder(self, new_weights: Optional[torch.Tensor]) -> None:
        # One 8-expert dot product creates one tap per sample.  The complete
        # kernel becomes active only after all 2,048 taps have been built.
        if self._build_weights is not None:
            index = self._build_index
            self._building_kernel[index] = torch.dot(
                self._build_weights, self.fir_kernels[:, 0, index]
            )
            self._build_index += 1
            if self._build_index == FIR_TAPS:
                self._active_kernel.copy_(self._building_kernel)
                self._build_weights = None
                self._build_index = 0
        if new_weights is not None:
            if self._build_weights is not None:
                raise RuntimeError("A mixed FIR kernel build missed its deadline.")
            self._build_weights = new_weights
            self._build_index = 0

    def process_sample(
        self, reference_sample: float, previous_error_sample: float
    ) -> float:
        self._sample_tensor.fill_(float(reference_sample))
        self._error_tensor.fill_(float(previous_error_sample))
        wavenet_output = self.wavenet.forward_sample(self._sample_tensor)
        fir_output = self._fir_sample(self._sample_tensor[0, 0, 0])
        controller_output = 0.5 * wavenet_output + 0.5 * fir_output
        new_weights = self.classifier.forward_sample(
            self._sample_tensor, controller_output, self._error_tensor
        )
        self._advance_kernel_builder(new_weights)
        return float(controller_output[0, 0, 0].item())
