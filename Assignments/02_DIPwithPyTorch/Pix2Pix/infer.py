import os
import re
import argparse
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from train import tensor_to_image, save_images

def load_model(model : torch.nn.Module):
    checkpoints = os.listdir('checkpoints/')
    checkpoints.sort(key=lambda f: int("".join(filter(str.isdigit, f)))) 
    checkpoint = checkpoints[-1]
    m = re.compile(r'pix2pix_model_epoch_(\d+)\.pth').match(checkpoint)
    if m is None:
        raise(Exception(f"checkpoint {checkpoint} not found"))
    last_epoch = int(m.group(1))
    state = torch.load(f"checkpoints/{checkpoint}", weights_only=True)
    model.load_state_dict(state)
    return last_epoch

def infer(device, batch_size = 10, folder_name="test_results"):
    model = FullyConvNetwork().to(device)
    epoch = load_model(model)
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)

    test_dataset = FacadesDataset(list_file='test_list.txt')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    model.eval()
    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(test_loader):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            outputs = model(image_rgb)
            count = image_rgb.shape[0]
            for j in range(count):
                input_img_np = tensor_to_image(image_rgb[j])
                target_img_np = tensor_to_image(image_semantic[j])
                output_img_np = tensor_to_image(outputs[j])
                comparison = np.hstack((input_img_np, target_img_np, output_img_np))
                cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i * batch_size + j + 1}.png', comparison)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    infer(device)