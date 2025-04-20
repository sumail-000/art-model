# PixelMind-X

A deep learning framework for advanced image generation and manipulation.

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install the dependencies:
```bash
# For a minimal installation
pip install -r requirements_minimal.txt

# For full installation
pip install -r requirements.txt
```

## Testing the Installation

Run the test script to verify your installation:
```bash
python test_installation.py
```

## Generate Images

Use the `generate_image.py` script to create images with Stable Diffusion:

```bash
python generate_image.py --prompt "a beautiful landscape with mountains and a lake, photorealistic"
```

Available options:
- `--prompt`: Text description of the image (required)
- `--negative_prompt`: What to avoid in the image
- `--seed`: Set a seed for reproducible results
- `--guidance_scale`: Control image adherence to the prompt (default: 7.5)
- `--steps`: Number of denoising steps (default: 50)

## Project Structure

- `src/`: Main source code
  - `multimodal/`: Multimodal models and processors
  - `generative/`: Image generation models
  - `learning/`: Custom learning and training modules
  - `evaluation/`: Metrics and evaluation tools
  - `api/`: API interfaces
- `configs/`: Configuration files
- `docs/`: Documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details. 