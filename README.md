# EDGAN

Main latent-domain training code and the single pretrained encoder/decoder pair used in the aligned evaluation experiments.

## Included files

- `train_codec.py`: joint encoder/decoder/discriminator training entry point.
- `models/codec.py`: encoder, decoder, and training discriminator definitions.
- `attacks/noise_attacks.py`, `losses/watermark_losses.py`, and `utils/`: required training dependencies.
- `configs/default.yaml`: recorded configuration of the selected joint-training run, restricted to main-training settings and with a portable output directory.
- `checkpoints/codec_weights.pt`: one frozen inference checkpoint containing the selected encoder and main decoder.

The pretrained encoder is from joint-training step 190000. The main test decoder was subsequently trained for 10000 additional decoder-only steps (checkpoint step 200000). Its tensors are the exact ones used in the aligned evaluations. The main training entry point alone does not reproduce that subsequent decoder refinement.

This repository contains no image-attack-specific decoders, optimizer checkpoints, training logs, image datasets, experiment sweeps, or diffusion-model weights. The discriminator architecture is required by main training; pretrained discriminator weights are not included.

## Download

Git LFS is required for the pretrained checkpoint:

```bash
git lfs install
git clone https://github.com/rrrrsyx/EDGAN.git
cd EDGAN
git lfs pull
```

The repository is private; the GitHub account used for cloning must have access.

## Install

Use Python 3.10 or newer and install a PyTorch build suitable for your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Packaging and checkpoint loading were verified with Python 3.13.5 and PyTorch 2.6.0. Training samples random 128-bit watermarks and operates in latent space; it does not require an image dataset or a Stable Diffusion model.

## Main training

```bash
python train_codec.py --config configs/default.yaml
```

The recorded configuration uses batch size 1024 and 224000 total steps. Adjust the device or batch size for available hardware if starting a new run. The released encoder was selected at step 190000; reproducing a stochastic training run is not guaranteed to produce bit-identical weights.

```bash
python train_codec.py --config configs/default.yaml --device cuda --batch_size 32
```

Training outputs are written to timestamped folders under `outputs/codec/`, which is excluded from Git. Full checkpoints created by this trainer can be supplied to `--resume`; the released inference-only checkpoint has no optimizer or discriminator state and is intended for inference loading below.

## Load the pretrained pair

Run the following from the repository root:

```python
import torch
from models.codec import WatermarkEncoder, WatermarkDecoder

checkpoint = torch.load(
    "checkpoints/codec_weights.pt", map_location="cpu", weights_only=True
)
config = checkpoint["model_config"]
encoder = WatermarkEncoder(
    bit_dim=config["bit_dim"],
    latent_channels=config["latent_channels"],
    latent_size=config["latent_size"],
)
decoder = WatermarkDecoder(
    bit_dim=config["bit_dim"],
    latent_channels=config["latent_channels"],
    latent_size=config["latent_size"],
    base_channels=config["decoder_base_channels"],
    decoder_channels=config["decoder_channels"],
)
encoder.load_state_dict(checkpoint["encoder"], strict=True)
decoder.load_state_dict(checkpoint["decoder"], strict=True)
encoder.eval()
decoder.eval()

with torch.inference_mode():
    bits = torch.randint(0, 2, (4, 128)).float()
    latents = encoder(bits)              # [4, 4, 64, 64]
    logits = decoder(latents)            # [4, 128]
    recovered = (torch.sigmoid(logits) > 0.5).to(torch.uint8)
```

For image extraction, feed inverted image latents into the decoder. Stable Diffusion generation, image attacks, VAE encoding, and DDIM inversion are outside this minimal training-and-weights repository.

## Integrity

`checkpoints/manifest.json` records the exact checkpoint SHA-256 and source checkpoint hashes. The selected checkpoint contains only encoder/decoder tensor state, model configuration, and the two source hashes. No tensor values, precision, or model parameters were changed during packaging.
