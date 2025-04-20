# Pixel Art Model Training Guide

This guide explains how to train a Stable Diffusion model on a pixel art dataset.

## 1. Dataset Preparation

The dataset has been prepared using the `prepare_dataset_with_prompts.py` script, which:
- Found 2,000 pixel art images in the dataset
- Matched each image with its original prompt from the metadata
- Split the data into train/validation/test sets (80/10/10)
- Created a properly formatted dataset in `data/pixel_art_dataset/`

If you need to re-prepare the dataset, run:
```bash
python prepare_dataset_with_prompts.py
```

## 2. Installing Dependencies

Before training, you need to install the required packages. Run:
```bash
python install_training_deps.py
```

This will install the following libraries:
- PyTorch and torchvision
- Accelerate for distributed training
- Transformers and diffusers for the Stable Diffusion model
- Additional utilities for training

## 3. Training the Model

To train the model with default parameters, run:
```bash
python train_pixelart_model.py
```

### Training Parameters

You can customize the training with these command-line arguments:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset_dir` | Path to dataset | `data/pixel_art_dataset` |
| `--pretrained_model_name_or_path` | Base model to fine-tune | `runwayml/stable-diffusion-v1-5` |
| `--output_dir` | Where to save checkpoints | `output/pixelart_model` |
| `--train_batch_size` | Batch size | `4` |
| `--max_train_steps` | Total training steps | `10000` |
| `--learning_rate` | Learning rate | `1e-5` |
| `--mixed_precision` | Mixed precision mode | `no` |

### Example Custom Training Command

For faster training with a smaller batch size and mixed precision:
```bash
python train_pixelart_model.py --train_batch_size 2 --gradient_accumulation_steps 2 --mixed_precision fp16 --max_train_steps 5000
```

For more parameters and options, run:
```bash
python train_pixelart_model.py --help
```

## 4. Testing the Trained Model

After training completes, you can generate images using:

```bash
python test_pixelart_model.py --model_path output/pixelart_model/final_model --prompt "pixel art forest with a small cabin"
```

### Test Parameters

You can customize image generation with these arguments:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to trained model | Required |
| `--prompt` | Text prompt for generation | "pixel art landscape with mountains and a lake" |
| `--num_images` | Number of images to generate | `4` |
| `--guidance_scale` | How closely to follow prompt | `7.5` |
| `--num_inference_steps` | Denoising steps | `50` |
| `--show_images` | Display images with matplotlib | `False` |

### Example Test Command

To generate a character with a specific seed:
```bash
python test_pixelart_model.py --model_path output/pixelart_model/final_model --prompt "pixel art character, wizard with blue robe and staff" --seed 42 --show_images
```

## 5. Training Notes

- Training will save checkpoints at regular intervals in the output directory
- Sample images are generated during training to monitor progress
- Training metrics are logged and can be viewed with TensorBoard
- Use early checkpoints if the model starts to overfit
- The model only fine-tunes the UNet part, keeping the text encoder and VAE frozen

## Hardware Requirements

For optimal training:
- NVIDIA GPU with at least 8GB VRAM
- 16GB+ system RAM
- Training time depends on your hardware and the number of steps 