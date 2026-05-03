# Clustering Stego: Steganography via Neural Network Parameter Initialization

This repository contains the official implementation of the paper: **"Steganography via Neural Network Parameter Initialization with High Fidelity and Imperceptibility"**.

## Abstract

Clustering Stego is a novel steganographic method that embeds secret information into the initial parameters of neural networks. By leveraging a pre-trained clustering-based encoder-decoder pair, secret bits are mapped into small clusters of model parameters. This approach ensures high embedding capacity and excellent imperceptibility, as the modified parameters maintain their original statistical distributions. Furthermore, the embedded secrets exhibit high robustness against model fine-tuning and pruning, maintaining high fidelity for the primary task.

## Key Features

- **Broad Model Compatibility**: Supports a wide range of architectures including:
  - **Computer Vision**: AlexNet, VGG16, ResNet18, DenseNet121, ViT.
  - **Natural Language Processing**: LSTM, Transformers.
  - **Generative Models**: VAE, GAN, Diffusion Models.
- **High Fidelity**: Negligible impact on the carrier model's primary task performance (e.g., classification accuracy).
- **Strong Imperceptibility**: Maintains weight distributions (checked via KL-divergence and entropy), making it resistant to statistical analysis.
- **Robustness**: Utilizes BCH (Bose-Chaudhuri-Hocquenghem) error correction coding for reliable secret extraction even after model training.
- **Efficient Embedding/Extraction**: Fast processing using GPU-accelerated encoder/decoder networks.

## Environment Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- matplotlib
- bchlib (for error correction)

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib bchlib
```

## Project Structure

```text
clustering_stego/
├── clustering_stego.py      # Main steganography class (ClusteringStego)
├── model.py                 # Encoder and Decoder network architectures
├── train_coder.py           # Script to train the clustering encoder/decoder
├── model/                   # Pre-trained models and encoder/decoder weights
├── utils/                   # Utility functions
│   ├── get_data.py          # Data loading for various datasets (CIFAR, MNIST, etc.)
│   ├── init_function.py     # Initialization wrappers for different models
│   ├── function.py          # Core math and BCH encoding/decoding functions
│   └── train.py             # Training loops for carrier models
├── example_classifier_cv.py # Example: Embedding in CV classifiers
├── example_classifier_nlp.py# Example: Embedding in NLP models (LSTM/Transformer)
├── example_vae.py           # Example: Embedding in VAEs
├── example_gan.py           # Example: Embedding in GANs
└── example_diffusion.py     # Example: Embedding in Diffusion models
```

## Usage

### 1. Embedding Secret Information

To embed secret bits into a carrier model (e.g., ResNet18), use the `ClusteringStego` class. It will automatically identify suitable layers and modify their initial parameters.

```python
from clustering_stego import ClusteringStego
from utils import init_function
from torchvision import models

# 1. Define initialization function for the target model
init_func = init_function.init_resnet18

# 2. Initialize ClusteringStego
cs = ClusteringStego(init_func, alpha=16, BCH=True)

# 3. Load or create the carrier model
model = models.resnet18()
model.load_state_dict(torch.load("model/resnet18_init_original.pth"))

# 4. Encode (Embed) secrets
secret_bits, secret_bits_bch = cs.encode(model)
print(f"Embedded {secret_bits.numel()} bits.")
```

### 2. Training the Carrier Model

After embedding, you can train the model as usual. The secret information is preserved within the weights.

```python
# Standard PyTorch training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(epochs):
    train_one_epoch(model, train_loader, criterion, optimizer)
```

### 3. Extracting Secret Information

Secret bits can be extracted from the trained model weights at any time.

```python
# Extract secrets from the model
extracted_bits, extracted_bits_bch = cs.decode(model)

# Calculate Accuracy
acc = (secret_bits_bch == extracted_bits_bch).sum() / extracted_bits_bch.numel()
print(f"Extraction Accuracy (BCH): {acc:.4f}")
```

## Running Examples

You can run the provided example scripts to see the full workflow:

```bash
# Example for CV Classifier (ResNet18 on CIFAR-10)
python example_classifier_cv.py --model resnet18 --dataset cifar10 --alpha 16 --BCH True

# Example for NLP Model (LSTM on SST2)
python example_classifier_nlp.py --model lstm --dataset sst2

# Example for GAN
python example_gan.py
```

## Experiment Data Download

https://pan.quark.cn/s/319784668315?pwd=8yFY

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{wang2025steganography,
  title={Steganography via Neural Network Parameter Initialization with High Fidelity and Imperceptibility},
  author={Wang, Na and Xu, Chenyi and Cao, Fang and Huang, Lin and Wang, Wei and Qin, Chuan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```