import os
print("Current directory:", os.getcwd())
print("Contents:", os.listdir())

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from pathlib import Path
import time

class ArtDataset(Dataset):
   def __init__(self, folder_path, transform=None):
       self.folder_path = folder_path
       self.transform = transform
       self.image_files = [f for f in Path(folder_path).glob("*.jpg")]
       print(f"Loading dataset: Found {len(self.image_files)} images")
   
   def __len__(self):
       return len(self.image_files)
   
   def __getitem__(self, idx):
       img_path = self.image_files[idx]
       try:
           image = Image.open(img_path).convert('RGB')
           if self.transform:
               image = self.transform(image)
           return image
       except Exception as e:
           print(f"Error loading image {img_path}: {str(e)}")
           return torch.zeros((3, 64, 64))

class Generator(nn.Module):
   def __init__(self, latent_dim):
       super(Generator, self).__init__()
       self.main = nn.Sequential(
           nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
           nn.BatchNorm2d(512),
           nn.ReLU(True),
           nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
           nn.BatchNorm2d(256),
           nn.ReLU(True),
           nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
           nn.BatchNorm2d(128),
           nn.ReLU(True), 
           nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
           nn.BatchNorm2d(64),
           nn.ReLU(True),
           nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
           nn.Tanh()
       )

   def forward(self, x):
       return self.main(x)

class Discriminator(nn.Module):
   def __init__(self):
       super(Discriminator, self).__init__()
       self.main = nn.Sequential(
           nn.Conv2d(3, 64, 4, 2, 1, bias=False),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(64, 128, 4, 2, 1, bias=False),
           nn.BatchNorm2d(128),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(128, 256, 4, 2, 1, bias=False),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(256, 512, 4, 2, 1, bias=False),
           nn.BatchNorm2d(512),
           nn.LeakyReLU(0.2, inplace=True),
           nn.Conv2d(512, 1, 4, 1, 0, bias=False),
           nn.Sigmoid()
       )

   def forward(self, x):
       return self.main(x)

def save_checkpoint(generator, discriminator, epoch, path="checkpoints"):
   os.makedirs(path, exist_ok=True)
   torch.save({
       'generator_state_dict': generator.state_dict(),
       'discriminator_state_dict': discriminator.state_dict(),
       'epoch': epoch
   }, f"{path}/checkpoint_epoch_{epoch}.pt")
   print(f"Saved checkpoint for epoch {epoch}")

def train_art_gan(folder_path, num_epochs=100, batch_size=32, latent_dim=100, lr=0.0002, save_interval=10):
   print("\n🎨 Starting the AI training process...")
   
   transform = transforms.Compose([
       transforms.Resize(64),
       transforms.CenterCrop(64),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
   ])
   
   dataset = ArtDataset(folder_path, transform=transform)
   dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   
   generator = Generator(latent_dim).to(device)
   discriminator = Discriminator().to(device)
   
   criterion = nn.BCELoss()
   g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
   d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
   
   print(f"\n🎯 Starting training for {num_epochs} epochs...")
   start_time = time.time()
   
   try:
       for epoch in range(num_epochs):
           for i, real_images in enumerate(dataloader):
               batch_size = real_images.size(0)
               real_images = real_images.to(device)
               
               d_optimizer.zero_grad()
               label_real = torch.ones(batch_size, 1, 1, 1).to(device)
               label_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
               
               output_real = discriminator(real_images)
               d_loss_real = criterion(output_real, label_real)
               
               noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
               fake_images = generator(noise)
               output_fake = discriminator(fake_images.detach())
               d_loss_fake = criterion(output_fake, label_fake)
               
               d_loss = d_loss_real + d_loss_fake
               d_loss.backward()
               d_optimizer.step()
               
               g_optimizer.zero_grad()
               output_fake = discriminator(fake_images)
               g_loss = criterion(output_fake, label_real)
               g_loss.backward()
               g_optimizer.step()
               
               if i % 10 == 0:
                   elapsed = time.time() - start_time
                   print(f'⏱️  {elapsed:.1f}s | Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                         f'Loss_D: {d_loss.item():.4f} Loss_G: {g_loss.item():.4f}')
           
           if (epoch + 1) % save_interval == 0:
               save_checkpoint(generator, discriminator, epoch + 1)
   
   except KeyboardInterrupt:
       print("\n⚠️  Training interrupted! Saving final checkpoint...")
       save_checkpoint(generator, discriminator, epoch + 1, path="checkpoints/interrupted")
       return generator
   
   print("\n✨ Training complete!")
   return generator

def generate_art(generator, output_dir="generated_artwork", latent_dim=100, num_images=5):
   print(f"\n🎨 Generating {num_images} new artworks...")
   os.makedirs(output_dir, exist_ok=True)
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   generator.eval()
   
   with torch.no_grad():
       noise = torch.randn(num_images, latent_dim, 1, 1).to(device)
       generated_images = generator(noise)
       generated_images = (generated_images + 1) / 2
       
       transform = transforms.ToPILImage()
       for i, img_tensor in enumerate(generated_images):
           image = transform(img_tensor)
           save_path = os.path.join(output_dir, f"generated_artwork_{i+1}.png")
           image.save(save_path)
           print(f"Saved artwork {i+1} to: {save_path}")

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
