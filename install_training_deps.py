import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"Error: command failed with exit code {process.returncode}")
        sys.exit(1)

def install_packages():
    packages = [
        "torch==2.0.1",
        "torchvision==0.15.2",
        "accelerate==0.21.0",
        "transformers==4.30.2",
        "diffusers==0.19.3",
        "ftfy==6.1.1",
        "scipy==1.10.1",
        "safetensors==0.3.1",
        "bitsandbytes==0.40.2",  # Optional, for 8-bit Adam optimizer
        "tensorboard==2.13.0",
        "matplotlib==3.7.2",
        "tqdm==4.65.0",
        "pillow==10.0.0",
    ]
    
    # Join packages into a single string
    packages_str = " ".join(packages)
    
    # Install packages
    run_command(f"{sys.executable} -m pip install {packages_str}")
    
    print("All packages installed successfully!")

if __name__ == "__main__":
    print("Installing dependencies for pixel art model training...")
    install_packages() 