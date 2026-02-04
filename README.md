# UNet Inference Engine

A lightweight, modular PyTorch implementation for performing semantic segmentation inference using the U-Net architecture. This project is designed to easily load a pre-trained `.pth` weight file and generate prediction masks for new images.

## 📂 Project Structure

```text
.
├── models/             # U-Net architecture definition
├── inputs/             # Store your input images here
├── outputs/            # Generated masks will be saved here
├── weights/            # Store your trained .pth model here
├── predict.py          # Main inference script
└── requirements.txt    # Python dependencies