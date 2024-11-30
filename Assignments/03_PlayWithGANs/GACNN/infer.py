import os
import re
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GACNN import GACNN
from train import tensor_to_image, save_images

def load_model(generator_model : torch.nn.Module, discriminator_model : torch.nn.Module):
    checkpoints = os.listdir('checkpoints/')
    checkpoints.sort(key=lambda f: int("".join(filter(str.isdigit, f)))) 
    checkpoint = checkpoints[-1]
    m = re.compile(r'epoch_(\d+)').match(checkpoint)
    if m is None:
        raise(Exception(f"checkpoint {checkpoint} not found"))
    last_epoch = int(m.group(1))
    generator_state = torch.load(f"checkpoints/{checkpoint}/gen_model.pth", weights_only=True)
    discriminator_state = torch.load(f"checkpoints/{checkpoint}/dsc_model.pth", weights_only=True)
    generator_model.load_state_dict(generator_state)
    discriminator_model.load_state_dict(discriminator_state)
    return last_epoch

def infer(device, batch_size = 10, folder_name="test_results"):
    model = GACNN()
    generator_model = model.genertor_model.to(device)
    discriminator_model = model.discriminator_model.to(device)
    epoch = load_model(generator_model, discriminator_model)
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)

    test_dataset = FacadesDataset(list_file='test_list.txt')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    generator_model.eval()
    with torch.no_grad():
        for i, (image_rgb, image_semantic) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)
            outputs = generator_model(image_semantic)
            count = image_rgb.shape[0]
            for j in range(count):
                input_img_np = tensor_to_image(image_semantic[j])
                target_img_np = tensor_to_image(image_rgb[j])
                output_img_np = tensor_to_image(outputs[j])
                comparison = np.hstack((input_img_np, target_img_np, output_img_np))
                cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i * batch_size + j + 1}.png', comparison)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', action="store_true")
    device = torch.device('cpu')
    infer(device)