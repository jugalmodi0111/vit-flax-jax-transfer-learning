# ViT-Flax-JAX Transfer Learning: Vision Transformer Fine-Tuning Pipeline

A comprehensive Jupyter notebook demonstrating **Flax/JAX setup**, **repository trimming**, **dataset integration** (CIFAR-10 and ImageNet-1k), and **Vision Transformer (ViT) fine-tuning** using TensorFlow Hub models converted from JAX checkpoints via `jax2tf`.

## Overview

This project provides an end-to-end pipeline for:
- Setting up a minimal, runtime-only Flax environment with smart fallback handling
- Generating detailed keep/remove reports for repository cleanup
- Loading and preprocessing datasets via TensorFlow Datasets (TFDS)
- Fine-tuning pre-trained Vision Transformers on custom datasets with simple, well-explained code

## Key Features

### 1. **Smart Flax Environment Setup**
- Automatic editable install with fallback to source-path import
- Dependency management for JAX, Flax, TensorFlow, and Optax
- Comprehensive import verification across Flax submodules
- Works even without packaging metadata (pyproject.toml/setup.py)

### 2. **Repository Correlation & Trimming**
- Exhaustive path scanning with keep/remove decisions
- JSON and CSV reports for every file/folder with justifications
- Safe, guarded deletion with dry-run mode
- Preserves all runtime code while removing docs, tests, examples, CI assets

### 3. **Dataset Integration**
- **CIFAR-10**: Auto-download, lightweight (50k training images, 10 classes)
- **ImageNet-1k**: Manual setup support with clear TFDS instructions
- Preprocessing pipeline: resize to 224×224, light augmentation, batching, prefetching
- Compatible with ViT input requirements

### 4. **TF Hub Vision Transformer Fine-Tuning**
- Loads pre-trained ViT feature extractors from TensorFlow Hub
- Frozen backbone + trainable classifier head for fast convergence
- Optional ViT unfreezing for full fine-tuning
- Simple Keras workflow with clear explanations for each step

### 5. **Runnable Examples**
- Flax Linen MLP forward pass
- Parameter serialization roundtrip
- TrainState + Optax gradient descent demo
- NNX (next-gen Flax API) example with best-effort compatibility

## Project Structure

```
.
├── JAX2TF-ViT-FLAX.ipynb    # Main notebook (all-in-one pipeline)
├── README.md                 # This file
├── trim_report.json          # (Generated) Path-level keep/remove decisions
├── trim_report.csv           # (Generated) CSV version of report
├── trim_log.txt              # (Generated) Deletion log
└── flax/                     # Flax source repository (if cloned locally)
```

## Setup Instructions

### Prerequisites
- Python 3.8+ (tested with Anaconda/Miniconda)
- Jupyter Notebook or JupyterLab
- macOS, Linux, or WSL2 (Windows)

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/jugalmodi0111/vit-flax-jax-transfer-learning.git
   cd vit-flax-jax-transfer-learning
   ```

2. **Open the notebook**
   ```bash
   jupyter notebook JAX2TF-ViT-FLAX.ipynb
   ```

3. **Run environment setup cells**
   - The notebook installs dependencies automatically
   - No manual `pip install` needed; all handled in-notebook

### Required Dependencies
Installed automatically by the notebook:
- `jax`, `jaxlib` (JAX for array operations)
- `flax` (neural network library)
- `optax` (gradient-based optimization)
- `tensorflow`, `tensorflow-datasets` (data loading)
- `tensorflow-hub` (pre-trained ViT models)

## Usage Guide

### Quick Start: CIFAR-10 Fine-Tuning
Run cells sequentially:
1. **Environment Setup** → installs dependencies and imports Flax
2. **TFDS Utilities** → defines dataset loaders
3. **CIFAR-10 Demo** → downloads CIFAR-10 (happens automatically)
4. **TF Hub Setup** → loads a ViT feature extractor
5. **Model Build** → ViT → Dense(10) classifier
6. **Train** → fit for 3 epochs (adjustable)
7. **Evaluate** → validation accuracy

Expected result: **~65-75% accuracy** on CIFAR-10 validation with frozen ViT in 3 epochs.

### Advanced: ImageNet-1k Evaluation
1. **Download ImageNet-1k** manually (registration required)
2. **Set environment variable**:
   ```bash
   export TFDS_DATA_DIR="$HOME/tfds_data"
   ```
3. **Place tar files** per [TFDS instructions](https://www.tensorflow.org/datasets/catalog/imagenet2012#manual_download_instructions)
4. **Run ImageNet loader cell** → validates setup and loads batches

### Repository Trimming (Optional)
If you cloned the full Flax repo:
1. **Run correlation cell** → generates `trim_report.json` and `trim_report.csv`
2. **Review reports** → inspect keep/remove decisions
3. **Set `APPLY_REMOVE = True`** in the apply cell
4. **Re-run** → removes non-runtime files (docs, tests, examples)

## Datasets

| Dataset | Rating | Citation | Usability |
|---------|--------|----------|-----------|
| **CIFAR-10** | ⭐⭐⭐⭐⭐ | Krizhevsky, 2009 | TFDS auto-download; 60k images (32×32); perfect for quick tests |
| **ImageNet-1k** | ⭐⭐⭐⭐⭐ | Russakovsky et al., 2015 | Manual TFDS setup; 1.28M training images; gold standard for ViT validation |

**CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html  
**ImageNet-1k**: https://image-net.org/challenges/LSVRC/2012/

## Model Architecture

```
Input (224×224×3 RGB image)
         ↓
ViT Feature Extractor (frozen)
  - Pre-trained on ImageNet-21k
  - Outputs: 768-dim embedding (ViT-B/16) or similar
         ↓
Dense(10, softmax) — trainable classifier
         ↓
Class probabilities (10 classes for CIFAR-10)
```

## Key Techniques

- **Transfer Learning**: Leverage ImageNet-pretrained ViT representations
- **Frozen Backbone**: Fast convergence by only training the classifier head
- **Optional Full Fine-Tuning**: Unfreeze ViT with lower learning rate for extra accuracy
- **Shape Polymorphism**: Models accept variable batch sizes via `jax2tf` conversion
- **TFDS Pipelines**: Efficient data loading with shuffle, map, batch, prefetch

## Performance Notes

### CIFAR-10 (Quick Demo)
- **Frozen ViT + Dense(10)**: ~65-75% in 3 epochs (50 steps/epoch)
- **Full epoch training**: ~80-85% with standard augmentation
- **Unfrozen ViT**: 85-90%+ with careful learning rate tuning

### ImageNet-1k (Validation Only)
- **TF Hub classifiers**: Match published accuracy (e.g., ViT-B/16: ~84.5% top-1)
- **Fine-tuned extractors**: Use for downstream tasks (feature extraction)

## Citations

If you use this pipeline, please cite the original works:

**Vision Transformer (ViT)**:
```
Dosovitskiy et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
ICLR 2021. https://arxiv.org/abs/2010.11929
```

**Flax**:
```
Heek et al. (2020). Flax: A neural network library and ecosystem for JAX.
http://github.com/google/flax
```

**JAX2TF Conversion**:
```
Paulu, S. (2021). ViT JAX to TensorFlow conversion.
https://github.com/sayakpaul/ViT-jax2tf
```

**CIFAR-10**:
```
Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images.
https://www.cs.toronto.edu/~kriz/cifar.html
```

**ImageNet**:
```
Russakovsky et al. (2015). ImageNet Large Scale Visual Recognition Challenge.
IJCV. https://doi.org/10.1007/s11263-015-0816-y
```

## Troubleshooting

### Flax Install Fails
- The notebook uses **source-path fallback** automatically
- No action needed; import will work from the local `flax/` directory

### TF Hub Model Download Hangs
- Check internet connection
- Try alternative handles from the list in the notebook
- Use a VPN if corporate firewall blocks tfhub.dev

### CIFAR-10 Not Downloading
- Ensure `tensorflow-datasets` is installed
- Check disk space (~170 MB required)
- Verify TFDS cache directory is writable (`~/tensorflow_datasets` by default)

### ImageNet TFDS Error
- Ensure manual tar files are placed correctly
- Set `TFDS_DATA_DIR` environment variable
- Follow [official TFDS guide](https://www.tensorflow.org/datasets/catalog/imagenet2012#manual_download_instructions)

## Contributing

Contributions welcome! To add features:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes with clear messages
4. Push and open a Pull Request

## License

This notebook is provided as-is for educational and research purposes. 

- **Flax**: Apache 2.0 License
- **JAX**: Apache 2.0 License
- **TensorFlow**: Apache 2.0 License
- **ViT models**: Vary by source; check TF Hub handles for specifics

Please respect dataset licenses (CIFAR-10: free for research; ImageNet: registration required).

## Acknowledgments

- Google Research for Flax, JAX, and Vision Transformer
- Sayak Paul for ViT JAX→TF conversion pipeline
- TensorFlow team for TFDS and TF Hub infrastructure
- Alex Krizhevsky for CIFAR-10
- ImageNet team for ILSVRC dataset

---

**Maintainer**: Jugal Modi ([@jugalmodi0111](https://github.com/jugalmodi0111))  
**Repository**: [vit-flax-jax-transfer-learning](https://github.com/jugalmodi0111/vit-flax-jax-transfer-learning)  
**Last Updated**: November 2025
# vit-flax-jax-transfer-learning
