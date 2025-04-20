import os
import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained PixelArt model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--output_dir", type=str, default="output/generated_images",
                        help="Directory to save generated images")
    parser.add_argument("--prompt", type=str, default="pixel art landscape with mountains and a lake",
                        help="Prompt to generate images from")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt to avoid certain elements")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="Guidance scale for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolution of generated images")
    parser.add_argument("--show_images", action="store_true",
                        help="Show images using matplotlib")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    # Use either CUDA or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipeline = pipeline.to(device)
    
    # Generate images
    print(f"Generating {args.num_images} images...")
    
    # Set up seed if provided
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f"Using seed: {args.seed}")
    
    # Generate images
    images = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        height=args.resolution,
        width=args.resolution,
        generator=generator
    ).images
    
    # Save and optionally display images
    print(f"Saving images to {args.output_dir}...")
    
    # Create a timestamp for filenames
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, image in enumerate(images):
        # Save image
        image_filename = f"{timestamp}_image_{i}.png"
        image_path = os.path.join(args.output_dir, image_filename)
        image.save(image_path)
        print(f"Saved image to {image_path}")
    
    # Show images if requested
    if args.show_images:
        plt.figure(figsize=(15, 15))
        rows = (args.num_images + 1) // 2  # Ceiling division
        cols = min(2, args.num_images)
        
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image)
            plt.axis("off")
        
        prompt_text = args.prompt
        if len(prompt_text) > 50:
            prompt_text = prompt_text[:47] + "..."
        plt.suptitle(f"Generated images for: {prompt_text}", fontsize=16)
        plt.tight_layout()
        plt.show()
    
    print("Done!")

if __name__ == "__main__":
    main() 