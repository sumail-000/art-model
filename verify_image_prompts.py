import os
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import glob
import textwrap

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

def match_and_display(df, image_files, image_names, num_samples=5):
    """Match images with their prompts and display them"""
    if df is None:
        print("No metadata available to match")
        return
    
    # Create a dictionary mapping image names to file paths
    image_map = {name: path for name, path in zip(image_names, image_files)}
    
    # Filter metadata to only include available images
    available_image_set = set(image_names)
    filtered_df = df[df['image_name'].isin(available_image_set)].copy()
    print(f"Matched {len(filtered_df)} images with their prompts")
    
    # Sample random images to display
    sampled_df = filtered_df.sample(num_samples)
    
    # Set up the figure
    fig = plt.figure(figsize=(15, 5 * num_samples))
    
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        image_name = row['image_name']
        prompt = row['prompt']
        image_path = image_map[image_name]
        
        # Display the image
        ax = fig.add_subplot(num_samples, 1, i+1)
        img = Image.open(image_path)
        ax.imshow(img)
        
        # Wrap the text for better display
        wrapped_prompt = textwrap.fill(prompt, width=80)
        ax.set_title(f"Image: {image_name}\nPrompt: {wrapped_prompt}", fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "verification_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "image_prompt_verification.png")
    plt.savefig(output_path)
    print(f"Saved verification image to {output_path}")
    
    # Also save a text file with more examples (using utf-8 encoding)
    try:
        text_samples = filtered_df.sample(20)
        text_output = os.path.join(output_dir, "prompt_verification.txt")
        
        # Using utf-8 encoding to handle special characters
        with open(text_output, 'w', encoding='utf-8') as f:
            for _, row in text_samples.iterrows():
                f.write(f"Image: {row['image_name']}\n")
                # Handle any problematic characters in prompts
                try:
                    prompt = row['prompt']
                    # Replace or remove problematic characters if needed
                    f.write(f"Prompt: {prompt}\n")
                except Exception as e:
                    f.write(f"Prompt: [Error displaying prompt: {str(e)}]\n")
                f.write("-" * 80 + "\n\n")
        print(f"Saved 20 prompt examples to {text_output}")
    except Exception as e:
        print(f"Error saving text examples: {str(e)}")
        
        # Fallback: Save in JSON format which handles Unicode better
        import json
        json_output = os.path.join(output_dir, "prompt_verification.json")
        json_data = []
        for _, row in text_samples.iterrows():
            json_data.append({
                "image_name": row['image_name'],
                "prompt": row['prompt']
            })
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Saved prompts as JSON to {json_output}")

def main():
    """Main function to verify image-prompt matches"""
    print("Verifying image-prompt matches")
    
    # Load metadata
    df = load_metadata()
    
    # Get available images
    image_files, image_names = get_available_images()
    
    if df is not None and image_files:
        # Match and display
        match_and_display(df, image_files, image_names)
    else:
        print("Could not verify matches. Please check the data files.")

if __name__ == "__main__":
    main() 