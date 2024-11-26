import torch
from torchvision import transforms
import numpy as np

def get_transforms(augment_prob=1.0):
    """Returns training and basic transforms"""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    basic_transform = transforms.ToTensor()
    
    return train_transform, basic_transform

def show_augmented_images(originals, augmenteds, title="Augmentation Examples"):
    """Display images as ASCII art in console"""
    print(f"\n{title}")
    print("-" * 80)
    
    for idx in range(3):
        # Convert tensors to numpy arrays and normalize to 0-1
        orig_img = originals[idx].squeeze().numpy()
        aug_img = augmenteds[idx].squeeze().numpy()
        
        # Resize to smaller dimensions for ASCII art
        small_orig = (orig_img[::2, ::2] > 0.5).astype(int)
        small_aug = (aug_img[::2, ::2] > 0.5).astype(int)
        
        # Convert to ASCII
        ascii_chars = [' ', '░', '▒', '▓', '█']
        
        print(f"\nPair {idx + 1}:")
        print("Original:")
        for row in small_orig:
            print(''.join(ascii_chars[int(val * 4)] for val in row))
        
        print("\nAugmented:")
        for row in small_aug:
            print(''.join(ascii_chars[int(val * 4)] for val in row))
        
        print("-" * 80)

def get_augmentation_examples(dataset, num_examples=3):
    """Get original and augmented image pairs"""
    original_images = []
    augmented_images = []
    
    aug_transform = transforms.Compose([
        transforms.RandomRotation(10),
    ])
    
    basic_transform = transforms.ToTensor()
    
    for i in range(num_examples):
        original_image = dataset[i][0]
        original_images.append(original_image)
        
        # Convert to PIL image for transforms
        pil_image = transforms.ToPILImage()(original_image)
        augmented_image = basic_transform(aug_transform(pil_image))
        augmented_images.append(augmented_image)
    
    return original_images, augmented_images 