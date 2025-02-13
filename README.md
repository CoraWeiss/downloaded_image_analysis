# downloaded_image_analysis
AI art analysis from an existing data file
# Art Generation GAN

A PyTorch implementation of a Generative Adversarial Network (GAN) for creating artistic images. This project uses deep learning to generate new artwork based on a training dataset of existing art pieces.

## Features

- Custom art dataset handling with automatic image preprocessing
- GAN architecture with Generator and Discriminator networks
- Checkpoint saving and loading functionality
- Art generation from trained models
- Progress tracking with real-time loss reporting
- Error handling and graceful interruption

## Requirements

- Python 3.x
- PyTorch
- torchvision
- Pillow (PIL)
- CUDA-capable GPU (optional, but recommended for faster training)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch torchvision pillow
```

## Project Structure

- `ArtDataset`: Custom dataset class for loading and preprocessing art images
- `Generator`: Neural network for generating new images
- `Discriminator`: Neural network for discriminating between real and generated images
- Training utilities and checkpoint management
- Art generation functionality

## Usage

### Training the Model

Place your training images in a directory (default: "part1") and run:

```python
python your_script.py
```

The script will:
1. Load and preprocess your training images
2. Train the GAN for the specified number of epochs
3. Save checkpoints at regular intervals
4. Generate sample artworks upon completion

### Configuration Options

You can modify these parameters in the script:
- `num_epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size for training (default: 32)
- `latent_dim`: Dimension of the latent space (default: 100)
- `lr`: Learning rate (default: 0.0002)
- `save_interval`: Checkpoint saving frequency (default: 10 epochs)

### Generating Art

The `generate_art` function can be used to create new artworks:

```python
generate_art(generator, output_dir="generated_artwork", num_images=10)
```

## Model Architecture

### Generator
- Takes random noise as input
- Uses transposed convolutions for upsampling
- Includes batch normalization and ReLU activations
- Outputs 64x64 RGB images

### Discriminator
- Processes 64x64 RGB images
- Uses convolutional layers
- Includes LeakyReLU activations and batch normalization
- Outputs probability of input being real

## Error Handling

The script includes robust error handling for:
- Image loading failures
- Training interruptions
- General exceptions

## Output

Generated artwork will be saved in the specified output directory (default: "generated_artwork") as PNG files.

## Checkpoints

Checkpoints are automatically saved:
- At specified intervals during training
- In the "checkpoints" directory
- Include both Generator and Discriminator states
- Can be used to resume training or generate art

## Performance Notes

- Training time varies based on dataset size and hardware
- GPU acceleration recommended for optimal performance
- Memory usage scales with batch size

## Contributing

Feel free to submit issues and enhancement requests!
