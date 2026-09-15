# Dataset format

Only the offline evaluation command, `python infer.py`, needs the following
local dataset layout. The streaming model does not read files at runtime: an
audio application supplies reference and previous-error samples directly.
The dataset is not distributed in this repository.

```text
data/CCF_dataset/
├── sh.npy
├── NOISE/
│   ├── <noise_name>.wav
│   └── ...
└── EXPECTED_NOISE/
    ├── <noise_name>_scene_01.wav
    ├── <noise_name>_scene_02.wav
    └── ...
```

`sh.npy` contains the secondary-path impulse responses. `infer.py` evaluates
10 paths by default: scenes 01--08 are the seen split and scenes 09--10 are
the unseen split. It uses the final two noise files in `NOISE/` as its default
held-out pair. The split indices, held-out count, segment duration, and start
offset can be changed through the command-line options of `infer.py`.

Do not commit audio, secondary-path measurements, or derived inference output
unless their redistribution rights are explicit.
