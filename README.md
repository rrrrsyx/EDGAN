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
