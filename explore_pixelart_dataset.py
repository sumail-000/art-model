import os
import pandas as pd
import json
from PIL import Image
import random
import shutil
from pathlib import Path
import glob

def get_available_images():
    """Get a list of all PNG files in the dataset directory structure"""
    image_files = []
    base_dir = "pixel_dataset/diffusiondb-pixelart/images"
    
    # Find all PNG files in the dataset directory
    for part_id in range(1, 3):
        part_dir = f"{base_dir}/part-{part_id:06d}"
        if os.path.exists(part_dir):
            png_files = glob.glob(os.path.join(part_dir, "*.png"))
            image_files.extend(png_files)
    
    print(f"Found {len(image_files)} image files in the dataset")
    return image_files

def format_dataset_from_files(image_files, output_dir="data/curated_pixel_art", limit=2000):
    """Format the dataset using available image files instead of metadata"""
    if not image_files:
        print("No image files found to process")
        return
    
    # Limit the total number of images to process
    if len(image_files) > limit:
        print(f"Limiting to {limit} images out of {len(image_files)}")
        random.shuffle(image_files)
        image_files = image_files[:limit]
    
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Prepare train, val, test splits (80/10/10)
    random.shuffle(image_files)
    n_total = len(image_files)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]
    
    # Create metadata for each split
    train_metadata = []
    val_metadata = []
    test_metadata = []
    
    # Process each split
    for split_name, split_files, split_metadata in [
        ("train", train_files, train_metadata),
        ("val", val_files, val_metadata),
        ("test", test_files, test_metadata)
    ]:
        print(f"Processing {split_name} split ({len(split_files)} images)")
        success_count = 0
        
        for source_path in split_files:
            # Extract filename
            image_name = os.path.basename(source_path)
            
            # Generate a generic prompt based on the filename
            # In a real scenario, you might want to create a better prompt or use an image captioning model
            prompt = f"pixel art image {Path(image_name).stem}"
            
            # Copy the image (rename with a more conventional name)
            new_name = f"{Path(image_name).stem}.png"
            dest_path = os.path.join(images_dir, new_name)
            
            try:
                shutil.copy(source_path, dest_path)
                
                # Add metadata entry
                split_metadata.append({
                    "image_path": f"images/{new_name}",
                    "description": prompt
                })
                success_count += 1
                
                # Print progress for large datasets
                if (len(split_files) > 500) and (success_count % 100 == 0) and (success_count > 0):
                    print(f"  Progress: {success_count}/{len(split_files)} images processed")
            except Exception as e:
                print(f"Error copying {source_path}: {e}")
    
    # Save metadata for each split
    for split_name, split_metadata in [
        ("train", train_metadata),
        ("val", val_metadata),
        ("test", test_metadata)
    ]:
        metadata_path = os.path.join(output_dir, f"{split_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        print(f"Saved {len(split_metadata)} records to {metadata_path}")
    
    print(f"\nDataset formatted for PixelMind-X in '{output_dir}'")

def create_sample_dataset(image_files, output_dir="sample_pixelart", n=50):
    """Create a sample dataset with images and simple prompts"""
    if not image_files:
        print("No image files found to create sample")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit the number of images
    sample_files = random.sample(image_files, min(n, len(image_files)))
    
    # Create a dictionary to store image-prompt pairs
    sample_data = {}
    success_count = 0
    
    # Copy images to the sample directory
    for source_path in sample_files:
        image_name = os.path.basename(source_path)
        prompt = f"pixel art image {Path(image_name).stem}"
        
        if os.path.exists(source_path):
            # Copy the image to the sample directory
            dest_path = os.path.join(output_dir, image_name)
            shutil.copy(source_path, dest_path)
            
            # Add to sample data
            sample_data[image_name] = {
                "prompt": prompt
            }
            success_count += 1
    
    # Save the sample metadata
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"\nCreated sample dataset with {success_count} images in '{output_dir}'")
    print(f"Sample metadata saved to '{output_dir}/metadata.json'")

def main():
    """Main function to explore the dataset and create formatted versions"""
    print("Exploring DiffusionDB Pixel Art Dataset with available files")
    
    # Get all available images
    image_files = get_available_images()
    
    if image_files:
        # Create a small sample dataset
        create_sample_dataset(image_files, n=50)
        
        # Format for PixelMind-X with all available images
        format_dataset_from_files(image_files, limit=2000)
    else:
        print("Could not find any image files. Please check the dataset structure.")

if __name__ == "__main__":
    main() 