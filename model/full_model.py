"""Inference-only architecture for ``ckpt/full_model_final.tar``.

This module contains only the architecture needed to load the released
checkpoint. Its module and parameter names match the released checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False,
                 dilation=1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=self.kernel_size, stride=stride,
            padding=0, bias=bias, dilation=self.dilation,
        )

    def forward(self, signal):
        return self.conv(F.pad(
            signal, (self.dilation * (self.kernel_size - 1), 0), mode="constant", value=0
        ))


class DilatedResidualBlock(nn.Module):
    def __init__(self, dilation, config):
        super().__init__()
        section = config["FeedforwardWaveNet"]["Resblock"]["conv1d"]
        self.res_ch = int(section["res"])
        self.skip_ch = int(section["skip"])
        self.conv1 = CausalConv1d(
            self.res_ch, 2 * self.res_ch, section["kernel"][0], bias=False,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            self.res_ch, self.res_ch + self.skip_ch, section["kernel"][1], bias=False,
        )

    def forward(self, signal):
        gated = self.conv1(signal)
        first, second = gated[:, :self.res_ch], gated[:, self.res_ch:2 * self.res_ch]
        output = self.conv2(torch.tanh(first) * torch.sigmoid(second))
        return output[:, :self.res_ch] + signal, output[:, self.res_ch:self.res_ch + self.skip_ch]


class FeedforwardWaveNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        section = config["FeedforwardWaveNet"]
        dilations = section["dilations"]
        self.dilations = (
            [2 ** index for index in range(int(dilations) + 1)]
            if isinstance(dilations, int) else list(dilations)
        )
        self.dilated_layers = nn.ModuleList(
            DilatedResidualBlock(dilation, config)
            for _ in range(int(section["num_stacks"]))
            for dilation in self.dilations
        )
        conv = section["conv"]
        self.conv0 = CausalConv1d(1, conv["out"][0], conv["kernel"][0], bias=False)
        self.conv1 = CausalConv1d(conv["input"][1], conv["out"][1], conv["kernel"][1], bias=False)
        self.conv2 = CausalConv1d(conv["input"][2], conv["out"][2], conv["kernel"][2], bias=False)
        self.conv3 = CausalConv1d(conv["input"][3], 1, conv["kernel"][3], bias=False)
        causal_layers = [self.conv0, self.conv1]
        for layer in self.dilated_layers:
            causal_layers.extend((layer.conv1, layer.conv2))
        causal_layers.extend((self.conv2, self.conv3))
        self.receptive_field = 1 + sum(
            layer.dilation * (layer.kernel_size - 1) for layer in causal_layers
        )

    def forward(self, signal):
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)
        if signal.dim() != 3:
            raise ValueError("FeedforwardWaveNet expects [batch, time] or [batch, channel, time].")
        output = self.conv1(self.conv0(signal))
        skips = []
        for layer in self.dilated_layers:
            output, skip = layer(output)
            skips.append(skip)
        output = torch.tanh(torch.stack(skips, dim=0).sum(dim=0))
        output = torch.tanh(self.conv2(output))
        return self.conv3(output).squeeze(1)


class _PerTimeChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(int(channels))

    def forward(self, signal):
        return self.norm(signal.transpose(1, 2)).transpose(1, 2)


class _CausalDilatedConvStage(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.left_padding = int(dilation) * (int(kernel_size) - 1)
        self.conv = nn.Conv1d(
            int(in_channels), int(out_channels), kernel_size=int(kernel_size), stride=1,
            dilation=int(dilation),
        )
        self.norm = _PerTimeChannelNorm(out_channels)
        self.activation = nn.SiLU()

    def forward(self, signal):
        return self.activation(self.norm(self.conv(F.pad(signal, (self.left_padding, 0)))))


class _GatingClassifier(nn.Module):
    def __init__(self, num_experts, channels, head_channels, temperature):
        super().__init__()
        self.temperature = float(temperature)
        channels = tuple(int(value) for value in channels)
        if len(channels) != 4:
            raise ValueError("gating_channels must contain four values.")
        stage_channels = (3,) + channels
        self.stages = nn.ModuleList(
            _CausalDilatedConvStage(stage_channels[index], stage_channels[index + 1], 3, dilation)
            for index, dilation in enumerate((1, 3, 9, 27))
        )
        self.head = nn.Sequential(
            nn.Linear(channels[-1] + 3, int(head_channels)), nn.SiLU(),
            nn.Linear(int(head_channels), int(num_experts)),
        )

    @staticmethod
    def _log_rms(feedback):
        return torch.log(torch.sqrt(feedback.square().mean(dim=-1) + 1e-8) + 1e-8).clamp(
            min=-12.0, max=12.0
        )

    def logits(self, reference, controller_output, error_signal):
        if reference.dim() != 2 or reference.shape != controller_output.shape or reference.shape != error_signal.shape:
            raise ValueError("Gating inputs must have matching [batch, time] shapes.")
        feedback = torch.stack((reference, controller_output, error_signal), dim=1)
        encoded = feedback
        for stage in self.stages:
            encoded = stage(encoded)
        return self.head(torch.cat((encoded.mean(dim=-1), self._log_rms(feedback)), dim=1))


class FullANCModel(nn.Module):
    """Complete checkpoint architecture used by offline full-model inference."""

    def __init__(self, config, num_experts, fir_taps):
        super().__init__()
        model_cfg = config["model"]
        self.num_experts = int(num_experts)
        self.fir_taps = int(fir_taps)
        self.wavenet_gain = float(model_cfg["feedforward_gain"])
        self.fir_gain = float(model_cfg["filter_experts_gain"])
        self.wavenet = FeedforwardWaveNet(config)
        required_context = self.wavenet.receptive_field - 1
        self.wavenet_context_samples = int(model_cfg.get("feedforward_context_samples", required_context))
        if self.wavenet_context_samples < required_context:
            raise ValueError("feedforward_context_samples is shorter than the WaveNet receptive field.")
        self.register_buffer("fir_kernels", torch.zeros(self.num_experts, 1, self.fir_taps))
        self.register_buffer("fir_biases", torch.zeros(self.num_experts))
        self.classifier = _GatingClassifier(
            self.num_experts,
            model_cfg["gating_channels"],
            model_cfg["gating_head_channels"],
            model_cfg["gating_temperature"],
        )

    @torch.no_grad()
    def compute_frozen_wavenet_chunk(self, reference, start, end):
        window_start = max(0, int(start) - self.wavenet_context_samples)
        output = self.wavenet(reference[:, window_start:int(end)])
        offset = int(start) - window_start
        return output[:, offset:offset + int(end) - int(start)]

    def forward(self, wavenet_output, reference, reference_history, feedback_reference,
                feedback_controller_output, feedback_error):
        logits = self.classifier.logits(
            feedback_reference, feedback_controller_output, feedback_error
        )
        weights = torch.softmax(logits / self.classifier.temperature, dim=1)
        kernels = torch.einsum("be,eck->bck", weights, self.fir_kernels)
        biases = torch.einsum("be,e->b", weights, self.fir_biases)
        batch = reference.shape[0]
        history_size = self.fir_taps - 1
        if reference_history is None:
            reference_history = reference.new_zeros(batch, history_size)
        extended = torch.cat((reference_history, reference), dim=1)
        fir_output = F.conv1d(
            extended.view(1, batch, -1), kernels, bias=biases, groups=batch
        ).squeeze(0)
        next_history = extended[:, -history_size:].detach() if history_size else extended[:, :0]
        controller_output = self.wavenet_gain * wavenet_output + self.fir_gain * fir_output
        return controller_output, weights, next_history
