import argparse
import logging
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
from models.unet_model import UNet


def preprocess_image(pil_image, scale=1.0):
    """
    Resize and normalize the input image.
    Returns a tensor with shape [1, C, H, W]
    """
    w, h = pil_image.size
    newW, newH = int(scale * w), int(scale * h)

    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

    pil_image = pil_image.resize((newW, newH), resample=Image.BICUBIC)
    img_ndarray = np.asarray(pil_image)

    # Handle grayscale or 3-channel images
    if img_ndarray.ndim == 2:
        img_ndarray = img_ndarray[np.newaxis, ...]
    else:
        # HWC -> CHW
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    # Normalize to 0-1 range
    img_ndarray = img_ndarray / 255.0

    return torch.as_tensor(img_ndarray.copy()).float().unsqueeze(0)


def predict_img(net, full_img, device, scale_factor=1.0, out_threshold=0.5):
    """
    Run inference on a single image
    """
    net.eval()
    img = preprocess_image(full_img, scale=scale_factor)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        # Assuming binary segmentation (1 class).
        # If multiclass, use F.softmax(output, dim=1) and torch.argmax
        if net.n_classes == 1:
            probs = torch.sigmoid(output)[0]
        else:
            probs = F.softmax(output, dim=1)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(full_img.size[1]),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    # Binary thresholding
    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images using U-Net')
    parser.add_argument('--model', '-m', default='weights/checkpoint.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', required=True, help='Path to input image')
    parser.add_argument('--output', '-o', metavar='OUTPUT', default='outputs/result.png', help='Path to output mask')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Minimum probability to consider a pixel as positive')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 1. Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 2. Initialize Model (Adjust channels/classes based on your training)
    # n_channels=3 for RGB, n_classes=1 for binary mask
    net = UNet(n_channels=3, n_classes=1, bilinear=True)

    # 3. Load Weights
    logging.info(f'Loading model from {args.model}')
    try:
        net.load_state_dict(torch.load(args.model, map_location=device))
        net.to(device=device)
    except FileNotFoundError:
        logging.error("File not found! Please place your .pth file in the weights folder.")
        exit()

    # 4. Load Image
    logging.info(f'Predicting image {args.input} ...')
    img = Image.open(args.input)

    # 5. Predict
    mask = predict_img(net=net,
                       full_img=img,
                       scale_factor=args.scale,
                       out_threshold=args.threshold,
                       device=device)

    # 6. Save Result
    result = Image.fromarray((mask * 255).astype(np.uint8))
    result.save(args.output)
    logging.info(f'Mask saved to {args.output}')