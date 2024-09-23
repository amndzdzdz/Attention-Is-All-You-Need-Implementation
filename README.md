# Transformer Implementation

This repository contains a PyTorch implementation of the Transformer model, originally introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Vaswani et al. The Transformer model has since become the foundation for many state-of-the-art models in NLP and other domains, such as BERT, GPT, and others. I've used the wmt14 dataset to train a model that translates text from german to english.

## Model Overview

The Transformer model relies solely on attention mechanisms, replacing the need for recurrence or convolution in traditional sequence-to-sequence models. Key components include:

- **Multi-head Self-Attention**: To capture relationships between all tokens in a sequence.
- **Positional Encoding**: To provide information about the order of tokens.
- **Feedforward Networks**: Applied to each position independently after attention layers.
- **Layer Normalization**: Added after each sub-layer.

## Infos

- Custom implementation of the Transformer architecture from scratch.
- Supports just training
- Some parts of the code are not written efficiently
- Due to hardware limitations, the model was only trained on a small dataset of 3000 Samples to make sure the implementation was correct. The model did converge nicely.

## In Progress

- Code documentation
- Inference
- Code efficiency

## Installation

You can clone this repository and get going by doing:

```bash
git clone https://github.com/amndzdzdz/Attention-Is-All-You-Need-Implementation.git

pip install -r requirements.txt
