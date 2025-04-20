import os
import pandas as pd
import json
import random
import shutil
from pathlib import Path
import glob

def load_metadata():
    """Load the metadata from the parquet file"""
    metadata_path = "pixel_dataset/diffusiondb-pixelart/metadata.parquet"
    
    if os.path.exists(metadata_path):
        print(f"Loading metadata from {metadata_path}")
        df = pd.read_parquet(metadata_path)
        print(f"Loaded {len(df)} records from metadata")
        return df
    else:
        print(f"Metadata file not found at {metadata_path}")
        return None

def get_available_images():
    """Get a list of all PNG files in the dataset directory structure"""
    image_files = []
    image_names = []
    base_dir = "pixel_dataset/diffusiondb-pixelart/images"
    
    # Find all PNG files in the dataset directory
    for part_id in range(1, 3):
        part_dir = f"{base_dir}/part-{part_id:06d}"
        if os.path.exists(part_dir):
            png_files = glob.glob(os.path.join(part_dir, "*.png"))
            image_files.extend(png_files)
            image_names.extend([os.path.basename(f) for f in png_files])
    
    print(f"Found {len(image_files)} image files in the dataset")
    return image_files, image_names

def create_matched_dataset(df, image_files, image_names, output_dir="data/pixel_art_dataset"):
    """Create a dataset with images matched to their original prompts"""
    if df is None:
        print("No metadata available to match")
        return
    
    # Create a dictionary mapping image names to file paths
    image_map = {name: path for name, path in zip(image_names, image_files)}
    
    # Filter metadata to only include available images
    available_image_set = set(image_names)
    filtered_df = df[df['image_name'].isin(available_image_set)].copy()
    print(f"Matched {len(filtered_df)} images with their original prompts")
    
    # Create output directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Prepare train, val, test splits (80/10/10)
    filtered_df = filtered_df.sample(frac=1.0, random_state=42)  # Shuffle the data
    n_total = len(filtered_df)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    train_df = filtered_df.iloc[:n_train]
    val_df = filtered_df.iloc[n_train:n_train+n_val]
    test_df = filtered_df.iloc[n_train+n_val:]
    
    # Create metadata for each split
    train_metadata = []
    val_metadata = []
    test_metadata = []
    
    # Process each split
    for split_name, split_df, split_metadata in [
        ("train", train_df, train_metadata),
        ("val", val_df, val_metadata),
        ("test", test_df, test_metadata)
    ]:
        print(f"Processing {split_name} split ({len(split_df)} images)")
        success_count = 0
        
        for _, row in split_df.iterrows():
            image_name = row['image_name']
            prompt = row['prompt']
            
            # Get the source image path
            source_path = image_map.get(image_name)
            
            if source_path and os.path.exists(source_path):
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
                    if success_count % 100 == 0:
                        print(f"  Progress: {success_count}/{len(split_df)} images processed")
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
    
    print(f"\nDataset with original prompts formatted for PixelMind-X in '{output_dir}'")
    
    # Also save a sample of prompts for review
    sample_prompts = filtered_df.sample(min(20, len(filtered_df)))
    sample_path = os.path.join(output_dir, "sample_prompts.txt")
    with open(sample_path, 'w') as f:
        for _, row in sample_prompts.iterrows():
            f.write(f"Image: {row['image_name']}\n")
            f.write(f"Prompt: {row['prompt']}\n")
            f.write("-" * 50 + "\n")
    
    print(f"Saved sample prompts to {sample_path}")

def main():
    """Main function to prepare the dataset with original prompts"""
    print("Preparing pixel art dataset with original prompts")
    
    # Load metadata
    df = load_metadata()
    
    # Get available images
    image_files, image_names = get_available_images()
    
    if df is not None and image_files:
        # Create matched dataset
        create_matched_dataset(df, image_files, image_names)
    else:
        print("Could not prepare dataset. Please check the data files.")

if __name__ == "__main__":
    main() 