# Released model archive

This folder contains the single complete-model archive distributed with the
inference-only release:

```text
ckpt/
└── full_model_final.tar
```

`full_model_final.tar` contains the final complete controller checkpoint.
`infer.py` loads it directly, using the minimal architecture in `model/` and
the inference-only parameters in `inference_config.yaml`.

For real-time sample-by-sample deployment, use the separately exported state
dictionary at `streaming/model/full_model_streaming.pth` through
`streaming.infer:create_model`.

Do not write generated results into `ckpt/`; use a directory below `exp/`
instead. Use Git LFS or release assets if a model file exceeds your hosting
provider's size limit.
