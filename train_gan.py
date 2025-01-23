folder_path = "part1"  # Start with part1 
generator = train_art_gan(
    folder_path,
    num_epochs=100,
    batch_size=32
)
generate_art(generator, "generated_artwork", num_images=10)
