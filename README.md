# EDGAN

Training code and pretrained encoder/decoder.

## Setup

```bash
git lfs install
git clone https://github.com/rrrrsyx/EDGAN.git
cd EDGAN
git lfs pull
pip install -r requirements.txt
```

## Training

```bash
python train_codec.py --config configs/default.yaml
```

## Pretrained weights

[`checkpoints/codec_weights.pt`](checkpoints/codec_weights.pt) contains the encoder and decoder weights.

Run this CPU example from the repository root after downloading the weights with `git lfs pull`.

```python
import torch
from models.codec import WatermarkCodec

weights = torch.load(
    "checkpoints/codec_weights.pt", map_location="cpu", weights_only=True
)
config = weights["model_config"]
codec = WatermarkCodec(
    bit_dim=config["bit_dim"],
    latent_channels=config["latent_channels"],
    latent_size=config["latent_size"],
    decoder_base_channels=config["decoder_base_channels"],
    decoder_channels=config["decoder_channels"],
)
codec.encoder.load_state_dict(weights["encoder"], strict=True)
codec.decoder.load_state_dict(weights["decoder"], strict=True)
codec.eval()

with torch.inference_mode():
    bits = torch.randint(0, 2, (1, config["bit_dim"])).float()
    latents = codec.encoder(bits)  # [1, 4, 64, 64]
    logits = codec.decoder(latents)
    recovered_bits = (torch.sigmoid(logits) > 0.5).to(torch.uint8)  # [1, 128]

print(latents.shape, recovered_bits.shape)
```

For extraction from an image, pass its DDIM-inverted latent tensor to `codec.decoder` and apply the same bit threshold.
