# UARM
Official implementation of "Uncertainty-Aware Reward Modeling for Stable RLHF"

## Requirements

```bash
pip install -r requirements.txt
```

## Quick Start

You can run the following command to train the UARM model.

### Stage 1: Download preference data

Download the preference data from huggingface into `rawdata` directory.

```bash
python download.py --data_name hs
```

### Stage 2: Embedding Extraction

Extract embeddings from a pretrained LLM reward model. This produces safetensors files with `embeddings` (Tensor[N, D]) and `labels` (Tensor[N]) keys. This stage requires access to the pretrained model like `FsfairX-LLaMA3-RM-v0.1`. You can download it into `ckpt` directory from [here](https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1). And then run:

```bash
python data_prepare.py --data_name hs --subset train
python data_prepare.py --data_name hs --subset test
```

Expected output location: `./embeddings/normal/{model_name}_{data_name}_{train|test}.safetensors`

### Stage 3: Data Preparation

Simulate binary noise on the labels with ratio $\rho_{01}$ and $\rho_{10}$. This stage requires access to the Stage 2 safetensors files. Run:

```bash
python simulate.py --data_name hs
```
Expected output location: `./embeddings/clean/{model_name}_{data_name}_{train|test}.safetensors`

### Stage 4: UARM Training & Evaluation

```bash
# log into configs.py
python benchmark_cqr.py --data_name hs

# or pass it into the script
python benchmark_cqr.py --data_name hs --subset test
```

This trains the UARM model on the binary-noisy data and evaluates it on the test set.
