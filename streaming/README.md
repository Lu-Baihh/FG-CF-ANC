# Streaming inference

This directory contains the released runtime for strictly causal,
sample-by-sample ANC inference. The streaming weight is included at
`streaming/model/full_model_streaming.pth`.

Use the public interface from an audio loop:

```python
from streaming.infer import create_model

model = create_model(device="cpu")
model.reset()
controller_output = model.process_sample(reference, previous_error)
```

`streaming.infer:create_model` returns a stateful controller. It retains only
past samples, updates the Gating Network every 2,048 samples, and constructs
the next mixed 2,048-tap FIR kernel one coefficient per sample. This avoids a
large computation spike at a block boundary.

```text
streaming/
├── infer.py                     Public create_model/reset/process_sample API
├── code/streaming_model.py       Causal model implementation
└── model/full_model_streaming.pth
                                 Released streaming model weight
```
