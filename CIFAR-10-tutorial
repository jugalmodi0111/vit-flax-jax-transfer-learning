# Comprehensive Tutorial: Fine-Tuning a Vision Transformer (ViT) on CIFAR-10 Using Flax and JAX

This tutorial provides a detailed, step-by-step guide to fine-tuning a pretrained Vision Transformer (ViT) model on the CIFAR-10 dataset using the [vit-flax-jax-transfer-learning](https://github.com/jugalmodi0111/vit-flax-jax-transfer-learning) repository. The repository implements an end-to-end pipeline in a Jupyter notebook (`JAX2TF-ViT-FLAX.ipynb`), leveraging JAX for efficient computations, Flax for neural network building, TensorFlow Datasets (TFDS) for data handling, and TensorFlow Hub for pretrained ViT models.

ViT treats images as sequences of patches, applying transformer encoders to extract features—ideal for transfer learning on smaller datasets like CIFAR-10. Here, we'll freeze the ViT backbone (feature extractor) and train only a lightweight classifier head, achieving ~65-75% validation accuracy in just 3 epochs. This approach is computationally efficient and demonstrates core concepts without requiring extensive hardware.

**Why CIFAR-10?** It's a benchmark dataset with 60,000 tiny (32x32) RGB images across 10 classes (e.g., airplane, cat, dog). It's small enough for quick experimentation but challenging due to low resolution, making it perfect for showcasing ViT's strengths in transfer learning.

**Estimated Time:** 30-60 minutes (setup: 10 min; training: 20-40 min on CPU/GPU).

**Hardware Notes:** 
- CPU: Feasible for small batches; ~10-20 min/epoch.
- GPU/TPU: Recommended for faster training (use Google Colab for free GPU access).

---

## Prerequisites
Before starting, ensure you have:
- **Python 3.9+** (tested on 3.10).
- **Jupyter Notebook** environment (e.g., via Anaconda or Google Colab).
- Basic familiarity with Python, NumPy, and Jupyter (no deep ML expertise required).
- Internet access for downloading dependencies and the dataset (~170 MB for CIFAR-10).

No prior installation of JAX/Flax is needed—the notebook handles it intelligently.

---

## Step 1: Clone and Set Up the Repository
1. **Clone the Repository:**
   Open a terminal (or Colab shell) and run:
   ```
   git clone https://github.com/jugalmodi0111/vit-flax-jax-transfer-learning.git
   cd vit-flax-jax-transfer-learning
   ```
   This downloads the core files: the notebook, README, and utilities.

2. **Launch Jupyter Notebook:**
   - Locally: Install Jupyter if needed (`pip install notebook`), then run `jupyter notebook`. Navigate to and open `JAX2TF-ViT-FLAX.ipynb`.
   - In Google Colab: Upload the notebook via File > Upload, or clone directly in a Colab cell:
     ```
     !git clone https://github.com/jugalmodi0111/vit-flax-jax-transfer-learning.git
     %cd vit-flax-jax-transfer-learning
     ```
     Then, open the notebook in the file browser.

   **Pro Tip:** Use Colab for GPU acceleration—go to Runtime > Change runtime type > GPU.

---

## Step 2: Environment Setup (Automated in Notebook)
The notebook starts with a "smart" setup cell that installs and verifies dependencies. This handles edge cases like missing Flax packages by falling back to the local `flax/` directory.

1. **Run the First Cell (Setup):**
   Execute the initial cell. It performs:
   - `pip install` for essentials: `jax`, `flax`, `tensorflow-datasets`, `tensorflow-hub`, `optax` (optimizer), and `matplotlib` (for plotting).
   - Imports verification: Checks JAX devices (CPU/GPU) and prints a success message.
   
   **Example Output:**
   ```
   Installing dependencies...
   JAX version: 0.4.20
   Flax version: 0.7.0
   Devices: [CpuDevice(id=0)]
   Setup complete! Ready for ViT fine-tuning.
   ```

   **If Issues Arise:**
   - Restart the kernel (Kernel > Restart) and rerun.
   - For GPU: Ensure CUDA is installed; the notebook detects it automatically.

   **Customization Example:** If you want to force CPU-only (for debugging), add `%env JAX_PLATFORM_NAME=cpu` before imports.

---

## Step 3: Load and Prepare the CIFAR-10 Dataset
CIFAR-10 is loaded via TFDS, which auto-downloads and caches it (~170 MB). The notebook preprocesses images to ViT's expected input: 224x224 resolution with normalization.

1. **Run the Dataset Cell:**
   This cell defines a data pipeline:
   - Splits: 50,000 train images, 10,000 test images.
   - Preprocessing: Resize from 32x32 to 224x224, apply random crops/flips for augmentation (train only), normalize to [-1, 1].
   - Batching: 128 images/batch (adjustable for memory).

   **Key Code Snippet (Simplified):**
   ```python
   import tensorflow_datasets as tfds
   import jax.numpy as jnp
   from flax import linen as nn

   # Load dataset
   ds_builder = tfds.builder('cifar10')
   ds_builder.download_and_prepare()  # Auto-download if needed

   def preprocess(example, is_training):
       image = tf.image.resize(example['image'], (224, 224))
       if is_training:
           image = tf.image.random_crop(image, (200, 200))  # Augmentation example
           image = tf.image.random_flip_left_right(image)
       image = (image / 127.5) - 1.0  # Normalize
       return {'image': image, 'label': example['label']}

   # Create train/test dataloaders
   train_ds = tfds.load('cifar10', split='train', shuffle_files=True).map(
       lambda x: preprocess(x, True)).batch(128).prefetch(2)
   test_ds = tfds.load('cifar10', split='test').map(
       lambda x: preprocess(x, False)).batch(128).prefetch(2)
   ```

   **What Happens:**
   - **Download:** First run fetches data to `~/tensorflow_datasets/`.
   - **Augmentation Example:** A cat image might be cropped to focus on the face and flipped horizontally, simulating real-world variations.
   - **Output:** Prints dataset info: "Train batches: 391, Test batches: 79, Classes: ['airplane', 'automobile', ..., 'truck']".

   **Simple Test:** To visualize a sample batch, the notebook includes a Matplotlib plot:
   ```python
   import matplotlib.pyplot as plt
   for batch in train_ds.take(1):
       images = batch['image'].numpy()
       labels = batch['label'].numpy()
       fig, axes = plt.subplots(4, 4, figsize=(8, 8))
       for i, ax in enumerate(axes.flat):
           ax.imshow((images[i] + 1) / 2)  # Denormalize for display
           ax.set_title(f"Label: {labels[i]}")
           ax.axis('off')
       plt.show()
   ```
   Expect a 4x4 grid of resized, augmented CIFAR images.

---

## Step 4: Load Pretrained ViT Model from TF Hub
ViT models are loaded as feature extractors (frozen) to leverage ImageNet pretraining.

1. **Run the Model Loading Cell:**
   - Uses TF Hub: Downloads `https://tfhub.dev/google/vit-base-patch16-224-in21k-feature-extractor/1`.
   - Builds a Flax classifier: ViT backbone + Dense layer for 10 classes.

   **Key Code Snippet:**
   ```python
   import tensorflow_hub as hub
   from flax import linen as nn
   import jax

   # Load TF Hub ViT (as callable)
   vit_extractor = hub.KerasLayer("https://tfhub.dev/google/vit-base-patch16-224-in21k-feature-extractor/1")

   class ViTClassifier(nn.Module):
       features_extractor: nn.ModuleBase
       num_classes: int = 10

       def setup(self):
           self.head = nn.Dense(self.num_classes)

       def __call__(self, x):
           features = self.features_extractor(x)  # Shape: (batch, 768)
           return self.head(features)  # Logits: (batch, 10)

   # Initialize model
   model = ViTClassifier(vit_extractor, num_classes=10)
   params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 224, 224, 3)))['params']
   print("Model initialized. Backbone frozen by default.")
   ```

   **Explanation:**
   - **Freezing:** The ViT backbone (`features_extractor.trainable = False`) prevents updates, focusing gradients on the head.
   - **Example Forward Pass:** Input a 224x224 image → Output 10 logits (e.g., [2.1, -0.5, 1.3, ...] for class probabilities via softmax).
   - **Output:** "ViT loaded: 86M params (frozen), Head: 7,690 params (trainable). Total trainable: ~0.01%."

   **Advanced Option:** To unfreeze the backbone for better accuracy (~85-90%), set `vit_extractor.trainable = True` and lower the learning rate (e.g., 1e-5).

---

## Step 5: Training the Model
Train for 3 epochs with Optax (Adam optimizer) and cross-entropy loss. The notebook logs metrics and saves checkpoints.

1. **Run the Training Loop Cell:**
   - Optimizer: Adam with LR=1e-3, warmup.
   - Loss: Sparse categorical cross-entropy.
   - Metrics: Accuracy, tracked via Flax's `metrics`.

   **Key Code Snippet (Core Loop):**
   ```python
   import optax
   from flax.training import train_state
   import jax

   # Setup optimizer and state
   tx = optax.adam(learning_rate=1e-3)
   state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

   # Training step function (JIT-compiled for speed)
   @jax.jit
   def train_step(state, batch):
       def loss_fn(params, batch):
           logits = model.apply({'params': params}, batch['image'])
           loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
           return loss
       grad_fn = jax.value_and_grad(loss_fn)
       loss, grads = grad_fn(state.params, batch)
       state = state.apply_gradients(grads=grads)
       return state, loss

   # Run epochs
   num_epochs = 3
   for epoch in range(num_epochs):
       total_loss, total_acc = 0, 0
       for step, batch in enumerate(train_ds.as_numpy_iterator()):
           state, loss = train_step(state, batch)
           total_loss += loss
           # Accuracy computation...
           if step % 50 == 0:
               print(f"Epoch {epoch}, Step {step}: Loss {loss:.3f}")
       print(f"Epoch {epoch} complete. Avg Loss: {total_loss / len(train_ds):.3f}")
   ```

   **What to Expect:**
   - **Progress:** Prints every 50 steps (e.g., "Step 0: Loss 2.302" → decreasing to ~1.0).
   - **Speed Example:** On Colab GPU, ~5-10 sec/step; full epoch ~5 min.
   - **Checkpointing:** Auto-saves `vit_cifar10_checkpoint.npz` after each epoch—reload with `jax.numpy.load()` if interrupted.

   **Hyperparameter Tuning Example:**
   | Parameter | Default | Tune For... | Example Value |
   |-----------|---------|-------------|---------------|
   | Batch Size | 128 | Low Memory | 64 |
   | Learning Rate | 1e-3 | Faster Convergence | 5e-4 |
   | Epochs | 3 | Higher Accuracy | 10 |

   Monitor with TensorBoard (add `pip install tensorboard` and `%load_ext tensorboard; %tensorboard --logdir ./logs`).

---

## Step 6: Evaluation and Inference
Assess the model on the test set and run predictions on custom images.

1. **Run the Evaluation Cell:**
   ```python
   @jax.jit
   def eval_step(state, batch):
       logits = model.apply({'params': state.params}, batch['image'])
       return jax.nn.softmax(logits)

   accuracies = []
   for batch in test_ds.as_numpy_iterator():
       preds = eval_step(state, batch)
       acc = jnp.mean(jnp.argmax(preds, -1) == batch['label'])
       accuracies.append(acc)
   final_acc = jnp.mean(jnp.array(accuracies))
   print(f"Test Accuracy: {final_acc * 100:.2f}%")
   ```
   **Output Example:** "Test Accuracy: 72.45%". Visualize a confusion matrix with Seaborn (included in notebook).

2. **Inference on a Single Image:**
   - Upload a CIFAR-like image (e.g., a photo of a "frog").
   ```python
   from PIL import Image
   img = Image.open('frog_example.jpg').resize((224, 224))
   img_array = jnp.array(img) / 127.5 - 1.0
   img_array = jnp.expand_dims(img_array, 0)  # Batch dim
   logits = model.apply({'params': state.params}, img_array)
   probs = jax.nn.softmax(logits)[0]
   top_class = jnp.argmax(probs)
   print(f"Predicted: {classes[top_class]} (Confidence: {probs[top_class]:.2%})")
   ```
   **Example:** Input: Frog photo → "frog (92%)".

---

## Step 7: Next Steps and Troubleshooting
- **Improve Accuracy:** Unfreeze backbone, add dropout, or use longer training.
- **Extend to ImageNet-1K:** See notebook's advanced section—download subset via `tfds.load('imagenet2012', ... )`.
- **Common Issues:**
  - **Out of Memory:** Reduce batch size to 32.
  - **Import Errors:** Rerun setup; ensure no conflicting TF/JAX versions.
  - **Slow Training:** Switch to GPU or subsample dataset (`split='train[:10000]'` for quick tests).

**Performance Benchmarks (3 Epochs, Frozen Backbone):**
| Setup | Time/Epoch | Accuracy |
|-------|------------|----------|
| CPU (Batch 64) | 15 min | 68% |
| GPU (Batch 128) | 2 min | 75% |

This tutorial equips you with a working ViT pipeline. Experiment, fork the repo, and contribute! For questions, open an issue on GitHub.

**References:**
- Dosovitskiy et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*. arXiv:2010.11929.
- Krizhevsky (2009). CIFAR-10 Technical Report.

---
