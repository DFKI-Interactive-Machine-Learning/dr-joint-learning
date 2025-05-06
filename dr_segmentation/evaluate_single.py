#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from config import IMAGE_SIZE  # or just set IMAGE_SIZE = 512 here if needed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image_resized = transform(image)
    return image_resized.unsqueeze(0), image_resized  # return both for saving

def save_image(tensor_img, path):
    np_img = tensor_img.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    np_img = np.clip(np_img, 0, 255).astype(np.uint8)
    Image.fromarray(np_img).save(path)

def save_prediction(pred_mask, save_path):
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = (pred_mask * 255).astype(np.uint8)
    Image.fromarray(pred_mask).save(save_path)

def main(args):
    image_tensor, image_for_save = load_image(args.image_path)
    image_tensor = image_tensor.to(device)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
    )

    if os.path.isfile(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from {args.model}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {args.model}")

    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        pred_softmax = F.softmax(output, dim=1)
        pred_class = torch.argmax(pred_softmax, dim=1)

    os.makedirs(args.output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.image_path))[0]
    pred_path = os.path.join(args.output_dir, f'{base_filename}_prediction.png')
    input_path = os.path.join(args.output_dir, f'{base_filename}_input.png')

    save_image(image_for_save, input_path)
    save_prediction(pred_class, pred_path)

    print(f"Saved input image to {input_path}")
    print(f"Saved prediction to {pred_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./results/output', help='Directory to save predictions')
    args = parser.parse_args()
    main(args)
