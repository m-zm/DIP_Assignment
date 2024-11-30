import os
import re
from datetime import datetime
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GACNN import GACNN
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(generator_model, discriminator_model, dataloader, generator_optimizer, discriminator_optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    generator_model.train()
    discriminator_model.train()
    total_dloss = 0.0
    total_aloss = 0.0
    total_l1loss = 0.0

    for i, (image_rgb, image_semantic) in tqdm(enumerate(dataloader), desc="train", total=len(dataloader), leave=False):
        img_count = image_rgb.size(0)

        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # result label
        real_result = torch.tensor([[1.0]]*img_count).to(device)
        fake_result = torch.tensor([[0.0]]*img_count).to(device)

        # train generator
        # Zero the gradients
        generator_optimizer.zero_grad()

        # Forward pass
        image_output = generator_model(image_semantic)

        # loss
        adversarial_loss = nn.BCELoss()(discriminator_model(image_output), real_result)
        l1_loss = nn.L1Loss()(image_output, image_rgb)
        generator_loss = 100*l1_loss + 1*adversarial_loss
        total_aloss += adversarial_loss.item()
        total_l1loss += l1_loss.item()

        # backward and step
        generator_loss.backward()
        generator_optimizer.step()

        # train discriminator
        # Zero the grad
        discriminator_optimizer.zero_grad()

        # loss
        real_loss = nn.BCELoss()(discriminator_model(image_rgb), real_result)
        fake_loss = nn.BCELoss()(discriminator_model(image_output.detach()), fake_result)
        discriminator_loss = (real_loss + fake_loss)/2 # type: ignore
        total_dloss += discriminator_loss.item()

        # backward and step
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_semantic, image_rgb, image_output, 'train_results', epoch, num_images=dataloader.batch_size)

        # Print loss information
        tqdm.write(f'{datetime.now().ctime()}, Epoch [{epoch + 1: >4d}/{num_epochs}], Step [{i + 1: >2d}/{len(dataloader)}], dLoss: {discriminator_loss.item():.4e}, aLoss: {adversarial_loss.item():.4e}, l1Loss: {l1_loss.item():.4e}')
    # return loss

def validate(generator_model, discriminator_model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    generator_model.eval()
    discriminator_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in tqdm(enumerate(dataloader), desc="validate", total=len(dataloader), leave=False):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            image_output = generator_model(image_semantic)

            # # Compute the loss
            # adversarial_loss = nn.BCELoss()(discriminator_model(image_output), real_result)
            # l1_loss = nn.L1Loss()(image_output, image_semantic)
            # loss = criterion(image_output, image_semantic)
            # val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_semantic, image_rgb, image_output, 'val_results', epoch, num_images=dataloader.batch_size)

    # # Calculate average validation loss
    # avg_val_loss = val_loss / len(dataloader)
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main(batch_size = 20, use_checkpoint = False):
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = GACNN()
    generator_model = model.genertor_model.to(device)
    discriminator_model = model.discriminator_model.to(device)
    criterion = nn.L1Loss()
    generator_optimizer = optim.Adam(generator_model.parameters(), lr=0.0001, betas=(0.5, 0.999)) # type: ignore
    discriminator_optimizer = optim.Adam(discriminator_model.parameters(), lr=0.0001, betas=(0.5, 0.999)) # type: ignore

    # Add a learning rate scheduler for decay
    generator_scheduler = StepLR(generator_optimizer, step_size=200, gamma=0.2)
    discriminator_scheduler = StepLR(discriminator_optimizer, step_size=200, gamma=0.2)

    last_epoch = 0
    # if use_checkpoint:
    #     checkpoints = os.listdir('checkpoints/')
    #     checkpoints.sort(key=lambda f: int("".join(filter(str.isdigit, f)))) 
    #     checkpoint = checkpoints[-1]
    #     m = re.compile(r'pix2pix_model_epoch_(\d+)\.pth').match(checkpoint)
    #     if m is None:
    #         raise(Exception(f"checkpoint {checkpoint} not found"))
    #     last_epoch = int(m.group(1))
    #     state = torch.load(f"checkpoints/{checkpoint}", weights_only=True)
    #     model.load_state_dict(state)

    # Training loop
    num_epochs = 10000
    for epoch in tqdm(range(last_epoch, num_epochs)):
        train_one_epoch(generator_model, discriminator_model, train_loader, generator_optimizer, discriminator_optimizer, criterion, device, epoch, num_epochs)
        validate(generator_model, discriminator_model, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        generator_scheduler.step()
        discriminator_scheduler.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs(f'checkpoints/epoch_{epoch + 1}', exist_ok=True)
            torch.save(generator_model.state_dict(), f'checkpoints/epoch_{epoch + 1}/gen_model.pth')
            torch.save(discriminator_model.state_dict(), f'checkpoints/epoch_{epoch + 1}/dsc_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
    parser.add_argument("-c", "--checkpoint", action="store_true")
    args = parser.parse_args()
    main(args.batch_size, args.checkpoint)
