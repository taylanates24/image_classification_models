# Image Classification Models

This repository contains various image classification models.

## Features

- **Transfer Learning**: Implementation of transfer learning techniques to enhance model performance.
- **Custom Architectures**: Development of custom neural network architectures tailored for multi-channel image classification.
- **Pre-trained Models**: You can find the pretrained models from timm (huggingface) library.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Torch 2.1
- Torchvision 0.15
- Torch2trt 0.5.0
- NumPy
- OpenCV

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/taylanates24/image_classification_models.git
   ```

2. **Build the Docker image**:

   ```bash
   docker build -t image_classification -f Dockerfile .
   ```

3. **Create a Docker container**:

   ```bash
   docker run --gpus "device=0" --ipc host -it -v $(pwd):/workspace image_classification
   ```

