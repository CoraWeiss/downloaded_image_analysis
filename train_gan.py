import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import time

[Previous classes and functions...]

if __name__ == "__main__":
    print("🎨 Art Generation AI - Starting Training")
    folder_path = "part1"  # Updated to use part1 folder
    
    try:
        generator = train_art_gan(
            folder_path,
            num_epochs=100,
            batch_size=32
        )
        
        output_dir = "generated_artwork"
        generate_art(generator, output_dir, num_images=10)
        print(f"\n✅ Process complete! Check the '{output_dir}' folder for your generated artwork!")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
