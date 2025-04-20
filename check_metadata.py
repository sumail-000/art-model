import pandas as pd
import os
import glob

def check_parquet_file(parquet_path):
    """Check if the parquet file exists and examine its contents"""
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        return False
    
    try:
        print(f"Loading parquet file: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Successfully loaded parquet file with {len(df)} rows")
        
        # Check columns
        print(f"Columns in the parquet file: {', '.join(df.columns)}")
        
        # Display sample rows
        print("\nSample data (first 5 rows):")
        print(df.head())
        
        return df
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return False

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
    image_filenames = [os.path.basename(f) for f in image_files]
    return image_filenames

def check_image_links(df, image_filenames):
    """Check if the images in the parquet file exist in the dataset"""
    if 'image_name' not in df.columns:
        if 'filename' in df.columns:
            image_column = 'filename'
        else:
            print("Error: Could not find image name column in parquet file")
            return
    else:
        image_column = 'image_name'
    
    # Get all image names from the parquet file
    parquet_images = set(df[image_column].tolist())
    dataset_images = set(image_filenames)
    
    # Count matches
    matches = parquet_images.intersection(dataset_images)
    
    print(f"Images in parquet file: {len(parquet_images)}")
    print(f"Images in dataset: {len(dataset_images)}")
    print(f"Matching images: {len(matches)}")
    
    if len(matches) > 0:
        # Check a few matching examples
        print("\nSample matching images:")
        sample_matches = list(matches)[:5]
        for img in sample_matches:
            sample_row = df[df[image_column] == img].iloc[0]
            if 'prompt' in df.columns:
                prompt_col = 'prompt'
            elif 'text' in df.columns:
                prompt_col = 'text'
            else:
                prompt_col = None
            
            print(f"Image: {img}")
            if prompt_col:
                print(f"Prompt: {sample_row[prompt_col]}")
            print("-" * 50)

def main():
    # Check the new parquet file
    parquet_path = "0000.parquet"
    df = check_parquet_file(parquet_path)
    
    if df is not False:
        # Get available images
        image_filenames = get_available_images()
        
        # Check if parquet file links to images
        check_image_links(df, image_filenames)
    
    # Also check the original metadata file
    original_metadata = "pixel_dataset/diffusiondb-pixelart/metadata.parquet"
    if os.path.exists(original_metadata):
        print("\n\nChecking original metadata file:")
        df_original = check_parquet_file(original_metadata)
        if df_original is not False:
            check_image_links(df_original, image_filenames)

if __name__ == "__main__":
    main() 