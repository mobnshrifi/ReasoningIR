# ReasoningIR

This repository provides a PyTorch implementation of a recurrent image
restoration architecture that augments the [NAFNet](https://github.com/megvii-research/NAFNet)
backbone with a latent feedback loop inspired by the reasoning mechanism of
[RFR-Net](https://github.com/jingyuanli001/RFR-Inpainting).

## Features

- **Modular NAFNet backbone** implemented in `reasoning_ir.models.nafnet`.
- **Latent recurrent refinement** that performs multi-step restoration while
  only passing hidden states in the feature space, keeping the runtime overhead
  small.
- **Demo script** for running inference on a single image.
- **Unit tests** that validate the tensor shapes for both the vanilla and
  recurrent variants.

## Getting started

1. Install the dependencies (a recent version of PyTorch is required).

   ```bash
   pip install -r requirements.txt
   ```

2. Run the demonstration script on an input image:

   ```bash
   python scripts/demo_inference.py degraded.png restored.png --iterations 6
   ```

   The script automatically selects CUDA when available.

3. Integrate the module into your own training pipeline:

   ```python
   import torch
   from reasoning_ir import RecurrentNAFNet

   model = RecurrentNAFNet()
   degraded = torch.rand(1, 3, 256, 256)
   restored = model(degraded)  # shape: (1, 3, 256, 256)
   ```

## Project structure

```
reasoning_ir/
├── __init__.py
└── models/
    ├── nafnet.py          # Backbone implementation
    └── recurrent.py       # Latent feedback wrapper
scripts/
└── demo_inference.py      # Minimal inference example
tests/
└── test_recurrent_nafnet.py
```

## Testing

Run the unit tests with:

```bash
pytest
```

## License

The code is released for research purposes. Check the original NAFNet and
RFR-Net repositories for their respective licenses when using pretrained
weights or datasets from those projects.
