import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import lpips
import imageio
import json
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPImageProcessor
import torchvision.models as models
import signal
import sys
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from diffusers.utils import BaseOutput
import dataclasses
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import math

# === Configuration Parameters ===
DATA_ROOT = "/projects/RobotiXX/yangzhe/VITON/dataset/preprocessed_data_v2_train"
OUTPUT_DIR = "/scratch/ykong7/VITON/checkpoints_direct_concat_approach_withlpips_v2_without_decay"
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
BATCH_SIZE = 16  # Batch size per GPU
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 0  # Number of warmup epochs
IMAGE_SIZE = (512, 384)
SAVE_EVAL = True
EVAL_NUM_SAMPLES = 16
GRADIENT_ACCUMULATION_STEPS = 2  # Increase gradient accumulation steps, effectively expanding batch size
SEED = 42
MIXED_PRECISION = True  # Whether to use mixed precision training
LPIPS_WARMUP_EPOCHS = 100  # LPIPS warmup
USE_TEXT_PROMPT = False  # Don't use text prompts, use clothing image instead
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"  # CLIP model to use

class VITONAugmentation:
    """
    Data augmentation for VITON dataset that handles simultaneous and independent
    augmentations for clothing and UNet input conditions.
    """
    def __init__(
        self,
        hflip_prob=0.5,
        shift_prob=0.5,
        shift_limit=0.2,
        scale_prob=0.5,
        scale_limit=0.2,
        hsv_prob=0.5,
        hsv_limit=5.0,
        contrast_prob=0.5,
        contrast_limit=0.3
    ):
        self.hflip_prob = hflip_prob
        self.shift_prob = shift_prob
        self.shift_limit = shift_limit
        self.scale_prob = scale_prob
        self.scale_limit = scale_limit
        self.hsv_prob = hsv_prob
        self.hsv_limit = hsv_limit
        self.contrast_prob = contrast_prob
        self.contrast_limit = contrast_limit
    
    def __call__(self, sample):
        """
        Apply augmentations to the sample.
        
        Args:
            sample: Dictionary containing:
                - name: Sample name
                - pixel_values: Target image [-1, 1]
                - pose: Dense pose condition [-1, 1]
                - cloth: Clothing image [-1, 1]
                - agnostic: Agnostic map [-1, 1]
                - parse: Parsing map [0, 1]
                
        Returns:
            Augmented sample with the same keys
        """
        # Create a copy of the sample to avoid modifying the original
        augmented = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        
        # Get tensors for easy access
        cloth = augmented["cloth"]  # [C, H, W]
        pixel_values = augmented["pixel_values"]  # Target
        pose = augmented["pose"]  # Dense pose
        agnostic = augmented["agnostic"]  # Agnostic map
        
        # Create a list of tensors that will be treated as UNet input condition
        unet_inputs = [pixel_values, pose, agnostic]
        
        # 1. Horizontal flip - applied simultaneously
        if random.random() < self.hflip_prob:
            # Flip the cloth
            augmented["cloth"] = TF.hflip(cloth)
            
            # Flip all UNet inputs
            for i, tensor in enumerate(unet_inputs):
                unet_inputs[i] = TF.hflip(tensor)
        
        # Update the augmented dictionary with the modified tensors
        augmented["pixel_values"] = unet_inputs[0]
        augmented["pose"] = unet_inputs[1]
        augmented["agnostic"] = unet_inputs[2]
        
        # 2. Random shift - applied independently
        if random.random() < self.shift_prob:
            # Apply to cloth
            augmented["cloth"] = self._random_shift(cloth, self.shift_limit)
        
        # Apply shift independently to UNet inputs
        if random.random() < self.shift_prob:
            for i, tensor in enumerate(unet_inputs):
                unet_inputs[i] = self._random_shift(tensor, self.shift_limit)
            
            # Update the augmented dictionary
            augmented["pixel_values"] = unet_inputs[0]
            augmented["pose"] = unet_inputs[1]
            augmented["agnostic"] = unet_inputs[2]
        
        # 3. Random scale - applied independently
        if random.random() < self.scale_prob:
            # Apply to cloth
            augmented["cloth"] = self._random_scale(cloth, self.scale_limit)
        
        # Apply scale independently to UNet inputs
        if random.random() < self.scale_prob:
            for i, tensor in enumerate(unet_inputs):
                unet_inputs[i] = self._random_scale(tensor, self.scale_limit)
            
            # Update the augmented dictionary
            augmented["pixel_values"] = unet_inputs[0]
            augmented["pose"] = unet_inputs[1]
            augmented["agnostic"] = unet_inputs[2]
        
        # 4. HSV adjustments - applied simultaneously to cloth and pixel_values
        if random.random() < self.hsv_prob:
            # Get the adjustment values (same for both)
            h_shift, s_shift, v_shift = self._get_hsv_params(self.hsv_limit)
            
            # Apply to cloth
            cloth_01 = (cloth + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            cloth_hsv = self._adjust_hsv(cloth_01, h_shift, s_shift, v_shift)
            augmented["cloth"] = cloth_hsv * 2 - 1  # Convert back to [-1, 1]
            
            # Apply to pixel_values (target image)
            pixel_01 = (unet_inputs[0] + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            pixel_hsv = self._adjust_hsv(pixel_01, h_shift, s_shift, v_shift)
            augmented["pixel_values"] = pixel_hsv * 2 - 1  # Convert back to [-1, 1]
        
        # 5. Contrast adjustments - applied simultaneously to cloth and pixel_values
        if random.random() < self.contrast_prob:
            # Get the adjustment value (same for both)
            contrast_factor = self._get_contrast_factor(self.contrast_limit)
            
            # Apply to cloth
            cloth_01 = (augmented["cloth"] + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            cloth_contrast = self._adjust_contrast(cloth_01, contrast_factor)
            augmented["cloth"] = cloth_contrast * 2 - 1  # Convert back to [-1, 1]
            
            # Apply to pixel_values (target image)
            pixel_01 = (augmented["pixel_values"] + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            pixel_contrast = self._adjust_contrast(pixel_01, contrast_factor)
            augmented["pixel_values"] = pixel_contrast * 2 - 1  # Convert back to [-1, 1]
        
        return augmented
    
    def _random_shift(self, img, limit):
        """Apply random shift to the image"""
        h, w = img.shape[-2:]
        
        # Calculate shift amount in pixels
        shift_x = random.uniform(-limit, limit) * w
        shift_y = random.uniform(-limit, limit) * h
        
        # Create the affine grid for the shift
        theta = torch.tensor([
            [1, 0, shift_x / (w / 2)],
            [0, 1, shift_y / (h / 2)]
        ], dtype=torch.float32)
        
        # Expand dimensions to batch size 1
        theta = theta.unsqueeze(0)
        
        # Create the grid
        grid = F.affine_grid(theta, [1, img.shape[0], h, w], align_corners=False)
        
        # Apply the grid to the image
        shifted_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='reflection')
        
        return shifted_img.squeeze(0)
    
    def _random_scale(self, img, limit):
        """Apply random scale to the image"""
        h, w = img.shape[-2:]
        
        # Calculate scale factor
        scale = random.uniform(1 - limit, 1 + limit)
        
        # Create the affine grid for the scale
        theta = torch.tensor([
            [scale, 0, 0],
            [0, scale, 0]
        ], dtype=torch.float32)
        
        # Expand dimensions to batch size 1
        theta = theta.unsqueeze(0)
        
        # Create the grid
        grid = F.affine_grid(theta, [1, img.shape[0], h, w], align_corners=False)
        
        # Apply the grid to the image
        scaled_img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False, mode='bilinear', padding_mode='reflection')
        
        return scaled_img.squeeze(0)
    
    def _get_hsv_params(self, hsv_limit):
        """Generate random HSV adjustment parameters"""
        h_shift = random.uniform(-hsv_limit, hsv_limit)
        s_shift = random.uniform(-hsv_limit, hsv_limit)
        v_shift = random.uniform(-hsv_limit, hsv_limit)
        return h_shift, s_shift, v_shift
    
    def _adjust_hsv(self, img, h_shift, s_shift, v_shift):
        """
        Adjust HSV values of an image.
        
        Args:
            img: tensor in [0, 1] range
            h_shift: hue shift in degrees
            s_shift: saturation shift (percentage)
            v_shift: value shift (percentage)
        """
        # Convert to PIL Image for HSV adjustment
        img_pil = TF.to_pil_image(img)
        
        # Apply HSV adjustments
        img_pil = TF.adjust_hue(img_pil, h_shift / 360.0)  # TF expects hue in [-0.5, 0.5]
        img_pil = TF.adjust_saturation(img_pil, 1.0 + s_shift / 100.0)
        img_pil = TF.adjust_brightness(img_pil, 1.0 + v_shift / 100.0)
        
        # Convert back to tensor
        return TF.to_tensor(img_pil)
    
    def _get_contrast_factor(self, contrast_limit):
        """Generate random contrast factor"""
        return random.uniform(1 - contrast_limit, 1 + contrast_limit)
    
    def _adjust_contrast(self, img, factor):
        """Adjust contrast of an image"""
        return TF.adjust_contrast(img, factor)

class VITONDirectConcatDataset(Dataset):
    def __init__(self, root, use_cloth_agnostic=True):
        # self.pose_dir = os.path.join(root, "pose")
        self.pose_dir = os.path.join(root, "parse_pose")
        self.cloth_dir = os.path.join(root, "cloth")
        self.person_dir = os.path.join(root, "person")
        self.agnostic_dir = os.path.join(root, "agnostic")
        self.parse_dir = os.path.join(root, "parse")
        self.names = [f.replace(".pt", "") for f in os.listdir(self.pose_dir) if f.endswith(".pt")]
        self.use_cloth_agnostic = use_cloth_agnostic
        print(f"Dataset initialized with {len(self.names)} samples")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        pose = torch.load(os.path.join(self.pose_dir, f"{name}.pt"), weights_only=False)
        cloth = torch.load(os.path.join(self.cloth_dir, f"{name}.pt"), weights_only=False)
        person = torch.load(os.path.join(self.person_dir, f"{name}.pt"), weights_only=False)
        agnostic = torch.load(os.path.join(self.agnostic_dir, f"{name}.pt"), weights_only=False)
        parse = torch.load(os.path.join(self.parse_dir, f"{name}.pt"), weights_only=False)
        
        # Since original data is in [0,1] range, convert to [-1,1] range
        pose = pose * 2 - 1  # Dense pose condition
        cloth = cloth * 2 - 1  
        person = person * 2 - 1
        agnostic = agnostic * 2 - 1  # Agnostic map
        
        return {
            "name": name, 
            "pixel_values": person,  # Target image
            "pose": pose,           # Dense pose condition
            "cloth": cloth,         # Clothing image, used for conditional generation
            "agnostic": agnostic,   # Agnostic map
            "parse": parse,         # Parsing map (may be needed for mask generation)
        }

# Integration with VITONDirectConcatDataset
class AugmentedVITONDirectConcatDataset(VITONDirectConcatDataset):
    def __init__(self, root, use_cloth_agnostic=True, apply_augmentation=True):
        super().__init__(root, use_cloth_agnostic)
        self.apply_augmentation = apply_augmentation
        if apply_augmentation:
            self.augmentation = VITONAugmentation(
                hflip_prob=0.5,
                shift_prob=0.5,
                shift_limit=0.2,
                scale_prob=0.5,
                scale_limit=0.2,
                hsv_prob=0.5,
                hsv_limit=5.0,
                contrast_prob=0.5,
                contrast_limit=0.3
            )
    
    def __getitem__(self, idx):
        # Get the original sample
        sample = super().__getitem__(idx)
        
        # Apply augmentation if enabled
        if self.apply_augmentation and random.random() < 0.5:  # 50% chance to apply augmentation
            sample = self.augmentation(sample)
        
        return sample

# --- Simplified ClothEncoder without adapter but with projection ---
class SimplifiedClothEncoder(nn.Module):
    def __init__(self, clip_model_name=CLIP_MODEL_NAME, output_dim=768):
        super().__init__()
        
        print(f"Initializing SimplifiedClothEncoder with CLIP model: {clip_model_name}")
        # Load CLIP Vision Model as the backbone
        self.backbone = CLIPVisionModel.from_pretrained(clip_model_name)
        # Load the corresponding image processor
        self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_name)

        # Freeze the CLIP backbone
        self.backbone.requires_grad_(False)
        # Set to evaluation mode
        self.backbone.eval()
        
        # Get the feature dimension from CLIP's output
        self.feature_dim = self.backbone.config.hidden_size  # Should be 1024 for ViT-L/14
        
        # Add a projection layer to match UNet's cross-attention dimension (768)
        self.projection = nn.Linear(self.feature_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, cloth_images):
        """Extract features from cloth images.
        
        Args:
            cloth_images: Tensor of shape [B, 3, H, W] in [-1, 1] range
            
        Returns:
            cloth_features: Tensor of shape [B, output_dim]
        """
        # Convert tensors to PIL images for CLIP processor
        pil_images = [transforms.ToPILImage()((img + 1) / 2.0) for img in cloth_images]
        # Process images for CLIP
        processed_inputs = self.image_processor(images=pil_images, return_tensors="pt").to(cloth_images.device)
        pixel_values = processed_inputs['pixel_values']

        # Extract features using CLIP backbone
        self.backbone.eval()
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)

        # Use the [CLS] token's embedding from the last hidden state
        # Shape: [batch_size, hidden_dim] (e.g., [B, 1024])
        clip_features = outputs.last_hidden_state[:, 0]  # CLS token is at index 0
        
        # Project to output dimension
        projected_features = self.projection(clip_features)
        
        return projected_features


# === Helper functions ===
def setup(rank, world_size):
    """Set up distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set GPU device for current process
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed environment"""
    dist.destroy_process_group()


def save_images_for_inspection(batch, output_dir, prefix=""):
    """Save images for inspection"""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(2, len(batch["name"]))):
        name = batch["name"][i]
        # Convert back to [0,1] range then save
        save_image((batch["pose"][i] + 1) / 2, os.path.join(output_dir, f"{prefix}_{name}_pose.png"))
        save_image((batch["pixel_values"][i] + 1) / 2, os.path.join(output_dir, f"{prefix}_{name}_person.png"))
        save_image((batch["cloth"][i] + 1) / 2, os.path.join(output_dir, f"{prefix}_{name}_cloth.png"))
        save_image((batch["agnostic"][i] + 1) / 2, os.path.join(output_dir, f"{prefix}_{name}_agnostic.png"))


def set_requires_grad(model, requires_grad=True):
    """Set whether model requires gradient computation"""
    for param in model.parameters():
        param.requires_grad = requires_grad


# Load original UNet and modify the first layer
def load_and_modify_unet(pretrained_model_path, in_channels=12, device=None):
    """Load UNet and modify the first convolutional layer to accept in_channels input channels"""
    # Load original UNet
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    if device is not None:
        unet = unet.to(device)
    
    # Save original convolutional layer weights
    orig_conv_weight = unet.conv_in.weight.data.clone()
    orig_conv_bias = None
    if unet.conv_in.bias is not None:
        orig_conv_bias = unet.conv_in.bias.data.clone()
    
    # Recreate the first convolutional layer, but with more input channels
    new_conv = nn.Conv2d(
        in_channels=in_channels,  # New number of input channels
        out_channels=unet.conv_in.out_channels,
        kernel_size=unet.conv_in.kernel_size,
        stride=unet.conv_in.stride,
        padding=unet.conv_in.padding,
        bias=unet.conv_in.bias is not None
    ).to(device)
    
    # Copy original weights to the first 4 channels
    with torch.no_grad():
        new_conv.weight.data[:, :4, :, :] = orig_conv_weight
        new_conv.weight.data[:, 4:, :, :] = 0  # Set remaining channels to 0
        if orig_conv_bias is not None:
            new_conv.bias.data = orig_conv_bias
    
    # Replace the first convolutional layer
    unet.conv_in = new_conv
    
    return unet


def generate_image(
    unet, 
    cloth_encoder, 
    vae, 
    noise_scheduler, 
    agnostic_image, 
    pose, 
    cloth_image, 
    text_encoder=None, 
    tokenizer=None, 
    prompt="", 
    device=None, 
    num_inference_steps=100, 
    guidance_scale=7.5
):
    """Generate image using the new concat input approach"""
    if device is None:
        device = agnostic_image.device
    
    # Prepare inputs
    batch_size = agnostic_image.shape[0]
    
    # Get latent space shape
    height, width = agnostic_image.shape[-2:]
    latent_height = height // 8
    latent_width = width // 8
    
    # Generate pure noise for each sample (using standard normal distribution)
    latent = torch.randn(batch_size, 4, latent_height, latent_width, device=device)
    
    # Get clothing features
    cloth_features = cloth_encoder(cloth_image)
    
    # Prepare text condition or clothing condition
    if USE_TEXT_PROMPT and text_encoder is not None and tokenizer is not None:
        text_input = tokenizer(
            [prompt] * batch_size, 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            return_tensors="pt"
        ).to(device)
        encoder_hidden_states = text_encoder(**text_input).last_hidden_state
    else:
        # Expand cloth_features to shape expected by UNet [batch_size, seq_len, dim]
        # Here we assume UNet expects 77 tokens, each with dimension 768
        encoder_hidden_states = cloth_features.unsqueeze(1).expand(-1, 77, -1)
    
    # Prepare diffusion process
    scheduler = noise_scheduler
    # Explicitly set timesteps and move to correct device
    timesteps = torch.linspace(
        scheduler.config.num_train_timesteps - 1, 
        0, 
        num_inference_steps, 
        dtype=torch.long, 
        device=device
    )
    
    # Use random noise as starting point
    noisy_latent = latent
    
    # Get latent space for agnostic and pose
    with torch.no_grad():
        latent_agnostic = vae.encode(agnostic_image).latent_dist.sample() * 0.18215
        latent_pose = vae.encode(pose).latent_dist.sample() * 0.18215
    
    # Create unconditional embeddings (for classifier-free guidance)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        if USE_TEXT_PROMPT and text_encoder is not None and tokenizer is not None:
            # Use text encoder to create unconditional embeddings
            uncond_input = tokenizer(
                [""] * batch_size, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                return_tensors="pt"
            ).to(device)
            uncond_embeddings = text_encoder(**uncond_input).last_hidden_state
        else:
            # When not using text prompts, create all-zero embeddings for unconditional input
            uncond_embeddings = torch.zeros_like(encoder_hidden_states)
        
        # Combine conditional and unconditional embeddings
        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
    
    # Step-by-step denoising
    for i, t in enumerate(tqdm(timesteps)):
        # Expand timestep to batch size
        t_batch = t.repeat(batch_size).to(device)
        
        # Expand latent representation for classifier-free guidance
        latent_model_input = torch.cat([noisy_latent] * 2) if do_classifier_free_guidance else noisy_latent
        
        # Prepare concat input
        # 1. noisy_latent - [B, 4, H, W]
        # 2. latent_agnostic - [B, 4, H, W]
        # 3. latent_pose - [B, 4, H, W]
        
        if do_classifier_free_guidance:
            # Repeat each condition
            latent_agnostic_expanded = torch.cat([latent_agnostic] * 2)
            latent_pose_expanded = torch.cat([latent_pose] * 2)
            
            # Concatenate all inputs - now a total of 12 channels(4+4+4)
            concat_input = torch.cat([
                latent_model_input,         # Noisy image (zt)
                latent_agnostic_expanded,   # Latent agnostic map E(xa)
                latent_pose_expanded,       # Latent dense pose condition E(xp)
            ], dim=1)
        else:
            # Concatenate all inputs
            concat_input = torch.cat([
                latent_model_input,     # Noisy image (zt)
                latent_agnostic,        # Latent agnostic map E(xa)
                latent_pose,            # Latent dense pose condition E(xp)
            ], dim=1)
        
        # Get noise prediction through UNet
        noise_pred = unet(
            concat_input,
            t_batch.float(),  # Ensure timestep is float type
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]
        
        # Classifier-free guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Execute denoising step
        noisy_latent = scheduler.step(noise_pred, t, noisy_latent).prev_sample
    
    # Decode final latent representation
    with torch.no_grad():
        image = vae.decode(noisy_latent / 0.18215).sample
    
    # Convert to image space [-1, 1] -> [0, 1]
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    
    return image


def evaluate(
    rank, 
    unet, 
    cloth_encoder, 
    vae, 
    noise_scheduler, 
    val_loader, 
    epoch, 
    output_dir, 
    text_encoder=None, 
    tokenizer=None,
    eval_all=False,
):
    """Evaluate current model performance"""
    if rank != 0:  # Only perform evaluation on main process
        return
    
    print(f"[Rank {rank}] Starting evaluation...")
    # Switch to evaluation mode
    unet.eval()
    cloth_encoder.eval()
    
    for batch_idx, batch in tqdm(enumerate(val_loader)):
        if not eval_all and batch_idx > 0:
            break
        # Get validation batch
        batch = next(iter(val_loader))
        names = batch["name"]
        
        # Move data to device
        device = torch.device(f"cuda:{rank}")
        poses = batch["pose"].to(device)
        targets = batch["pixel_values"].to(device)
        agnostics = batch["agnostic"].to(device)
        cloth_imgs = batch["cloth"].to(device)
        
        # Generate images
        images = []
        gen_tensors = []
        with torch.no_grad():
            for i in range(len(names)):
                prompt = "" if not USE_TEXT_PROMPT else "a person wearing fashionable clothes"
                
                gen_img = generate_image(
                    unet=unet,
                    cloth_encoder=cloth_encoder,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    agnostic_image=agnostics[i:i+1],
                    pose=poses[i:i+1],
                    cloth_image=cloth_imgs[i:i+1],
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    device=device,
                    num_inference_steps=100,
                    guidance_scale=7.5
                )
                
                # Save generated results
                gen_tensor = gen_img[0].cpu()
                gen_tensors.append(gen_tensor)
                img_pil = transforms.ToPILImage()(gen_tensor)
                images.append(img_pil)
                print(f"[Rank {rank}] Generated sample {i+1}/{len(names)}")
    
    # Save evaluation results
    save_dir = os.path.join(output_dir, f"epoch_{epoch}_eval")
    os.makedirs(save_dir, exist_ok=True)
    
    gif_frames = []
    
    # Initialize LPIPS loss
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    total_lpips = 0.0
    sample_count = 0
    
    # Create directories
    GEN_DIR = os.path.join(save_dir, "generated")
    GT_DIR = os.path.join(save_dir, "groundtruth")
    COMPARE_DIR = os.path.join(save_dir, "comparison")
    CONTROLS_DIR = os.path.join(save_dir, "controls")
    
    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    os.makedirs(COMPARE_DIR, exist_ok=True)
    os.makedirs(CONTROLS_DIR, exist_ok=True)
    
    # Save results
    for i, (name, image) in enumerate(zip(names, images)):
        # Save generated image
        image.save(os.path.join(GEN_DIR, f"{name}.png"))
        
        # Save control inputs
        save_image((poses[i] + 1) / 2, os.path.join(CONTROLS_DIR, f"{name}_pose.png"))
        save_image((agnostics[i] + 1) / 2, os.path.join(CONTROLS_DIR, f"{name}_agnostic.png"))
        save_image((cloth_imgs[i] + 1) / 2, os.path.join(CONTROLS_DIR, f"{name}_cloth.png"))
        
        # Calculate LPIPS and save ground truth image
        # Ensure both tensors are on the same device
        pred_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)
        # Convert to [-1,1] range to match targets
        pred_tensor = pred_tensor * 2 - 1
        target_tensor = targets[i].unsqueeze(0)
        
        lpips_value = perceptual_loss_fn(pred_tensor, target_tensor).item()
        total_lpips += lpips_value
        sample_count += 1
        
        # Save ground truth image
        target_image = ((targets[i] + 1) / 2.0).clamp(0, 1)
        save_image(target_image, os.path.join(GT_DIR, f"{name}.png"))
        
        # Generate comparison image: control input | generated image | ground truth image
        pose_tensor = ((poses[i] + 1) / 2.0).cpu()
        cloth_tensor = ((cloth_imgs[i] + 1) / 2.0).cpu()
        agnostic_tensor = ((agnostics[i] + 1) / 2.0).cpu()
        gen_tensor = gen_tensors[i]
        target_tensor = target_image.cpu()
        
        # Create comparison image
        comparison = torch.cat([
            torch.cat([pose_tensor, cloth_tensor], dim=2),
            torch.cat([agnostic_tensor, gen_tensor], dim=2),
            torch.cat([gen_tensor, target_tensor], dim=2)
        ], dim=1)
        
        save_image(comparison, os.path.join(COMPARE_DIR, f"{name}_comparison.png"))
        gif_frames.append(np.array(transforms.ToPILImage()(comparison)) * 255)
    
    # Save GIF and evaluation metrics
    if gif_frames:
        imageio.mimsave(os.path.join(save_dir, "eval.gif"), [frame.astype(np.uint8) for frame in gif_frames], duration=0.5)
    
    # Save evaluation metrics
    if sample_count > 0:
        avg_lpips = total_lpips / sample_count
        with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
            f.write(f"LPIPS: {avg_lpips:.4f}\n")
    
    print(f"[Rank {rank}] Evaluation completed, LPIPS: {avg_lpips:.4f}")
    return avg_lpips


# Helper nullcontext for conditional execution with mixed precision training
class nullcontext:
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def load_checkpoint(checkpoint_path, unet, cloth_encoder, optimizer, lr_scheduler, dataloader, device):
    """Load model and training state from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load UNet weights
    unet_path = os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.bin")
    if os.path.exists(unet_path):
        unet_state_dict = torch.load(unet_path, map_location=device)
        unet.load_state_dict(unet_state_dict)
        print("Loaded UNet weights")
    else:
        print(f"Warning: UNet weights not found at {unet_path}")
    
    # Load cloth encoder weights
    cloth_encoder_path = os.path.join(checkpoint_path, "cloth_encoder.pth")
    if os.path.exists(cloth_encoder_path):
        cloth_encoder_state_dict = torch.load(cloth_encoder_path, map_location=device)
        cloth_encoder.load_state_dict(cloth_encoder_state_dict)
        print("Loaded cloth encoder weights")
    else:
        print(f"Warning: Cloth encoder weights not found at {cloth_encoder_path}")
    
    # Try to load training metadata
    training_info_path = os.path.join(checkpoint_path, "training_info.json")
    last_epoch = 0
    global_step = 0
    
    if os.path.exists(training_info_path):
        with open(training_info_path, "r") as f:
            training_info = json.load(f)
        
        last_epoch = training_info.get("epoch", 0)
        global_step = training_info.get("global_step", 0)
        print(f"Loaded training info: epoch={last_epoch}, global_step={global_step}")
    else:
        # Extract epoch number from checkpoint path
        epoch_info = os.path.basename(checkpoint_path)
        if epoch_info.startswith("epoch_"):
            last_epoch = int(epoch_info.replace("epoch_", ""))
            print(f"Resuming from epoch {last_epoch}")
        else:
            last_epoch = 0
            print("Could not determine last epoch, starting from 0")
        
        # Estimate global step
        global_step = last_epoch * (len(dataloader) // GRADIENT_ACCUMULATION_STEPS)
    
    # Reset and step the scheduler to the appropriate point
    for _ in range(global_step):
        lr_scheduler.step()
    
    print(f"Restored learning rate to {lr_scheduler.get_last_lr()[0]}")
    
    return last_epoch, global_step

def train(rank, world_size, args):
    """Run training process on specific GPU"""
    # Set up distributed training environment
    setup(rank, world_size)
    
    # Set random seed for reproducible results
    if SEED is not None:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
    
    # Create output directory
    if rank == 0:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Initialize tensorboard
        writer = SummaryWriter(log_dir=os.path.join(OUTPUT_DIR, "logs"))
    
    # Print distributed training status
    print(f"[Rank {rank}] GPU: {torch.cuda.get_device_name(rank)}")
    
    # Initialize dataset and data loaders
    dataset = AugmentedVITONDirectConcatDataset(DATA_ROOT, use_cloth_agnostic=True, apply_augmentation=True)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=SEED
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,  # Must be False when using DistributedSampler
        num_workers=4, 
        pin_memory=True,
        sampler=train_sampler
    )
    
    # Load validation dataset
    val_dataset = VITONDirectConcatDataset(
        DATA_ROOT.replace("preprocessed_data_v2_train", "preprocessed_data_v2_test"), 
        use_cloth_agnostic=True
    )
    
    val_loader = DataLoader(val_dataset, batch_size=EVAL_NUM_SAMPLES, shuffle=False)
    
    # Configure device
    device = torch.device(f"cuda:{rank}")
    
    # Load models
    print(f"[Rank {rank}] Starting to load models...")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)  # Keep VAE frozen
    
    # Load and modify UNet
    unet = load_and_modify_unet(PRETRAINED_MODEL, in_channels=12, device=device)
    
    # Get UNet cross-attention dimension
    unet_cross_attention_dim = unet.config.cross_attention_dim  # Usually 768
    
    # Load scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(PRETRAINED_MODEL, subfolder="scheduler")
    
    # If using text prompts, load text encoder, otherwise set to None
    if USE_TEXT_PROMPT:
        tokenizer = CLIPTokenizer.from_pretrained(PRETRAINED_MODEL, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(PRETRAINED_MODEL, subfolder="text_encoder")
        text_encoder = text_encoder.to(device)
        text_encoder.requires_grad_(False)  # Keep text encoder frozen
    else:
        tokenizer = None
        text_encoder = None
    
    # Create simplified cloth encoder, ensuring output dimension matches UNet cross-attention dimension
    cloth_encoder = SimplifiedClothEncoder(clip_model_name=CLIP_MODEL_NAME, output_dim=unet_cross_attention_dim)
    cloth_encoder = cloth_encoder.to(device)
    
    print(f"[Rank {rank}] Models loaded, UNet cross-attention dimension: {unet_cross_attention_dim}")
    
    # Set training states
    vae.eval()  # Keep VAE in evaluation mode
    if text_encoder is not None:
        text_encoder.eval()  # Keep text encoder in evaluation mode
    
    # Set UNet to training mode
    set_requires_grad(unet, True)
    
    # Set cloth encoder to training mode
    set_requires_grad(cloth_encoder, True)
    
    # Wrap models as DDP models
    unet = DDP(unet, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    cloth_encoder = DDP(cloth_encoder, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    # Optimizer parameter groups
    optimizer_grouped_parameters = [
        {"params": [p for p in unet.module.parameters() if p.requires_grad], "lr": LEARNING_RATE},
        {"params": [p for p in cloth_encoder.module.parameters() if p.requires_grad], "lr": LEARNING_RATE}
    ]
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=LEARNING_RATE,
        # weight_decay=0.01
    )
    
    # Calculate total training steps
    steps_per_epoch = len(dataloader) // GRADIENT_ACCUMULATION_STEPS
    total_train_steps = NUM_EPOCHS * steps_per_epoch
    num_warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    
    # Set up learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine", 
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=total_train_steps
    )
    
    # Mixed precision training setup
    scaler = torch.cuda.amp.GradScaler() if MIXED_PRECISION else None
    
    # Perceptual loss
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)

    # Initialize starting epoch and global step
    start_epoch = 0
    global_step = 0

    # Resume from checkpoint if specified
    if args.resume_from is not None:
        resume_checkpoint_path = os.path.join(OUTPUT_DIR, args.resume_from)
        if os.path.exists(resume_checkpoint_path):
            # Only load checkpoint on the main process
            if rank == 0:
                print(f"[Rank {rank}] Loading checkpoint from {resume_checkpoint_path}")
            
            # Load checkpoint before DDP wrapping
            temp_unet = unet.module
            temp_cloth_encoder = cloth_encoder.module
            
            # Load checkpoint
            start_epoch, global_step = load_checkpoint(
                resume_checkpoint_path,
                temp_unet,
                temp_cloth_encoder,
                optimizer,
                lr_scheduler,
                dataloader,
                device
            )
            
            # Start from the next epoch
            start_epoch += 1
            
            if rank == 0:
                print(f"[Rank {rank}] Resuming training from epoch {start_epoch}, global step {global_step}")
        else:
            print(f"[Rank {rank}] Warning: Checkpoint directory {resume_checkpoint_path} not found. Starting from scratch.")
    
    # Analyze and inspect dataset
    if rank == 0:
        sample_batch = next(iter(dataloader))
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor) and len(value.shape) > 1:
                print(f"{key} shape: {value.shape}, min: {value.min().item():.4f}, max: {value.max().item():.4f}")
        
        # Save sample images for inspection
        save_images_for_inspection(sample_batch, os.path.join(OUTPUT_DIR, "data_check"), "sample")
    
    # Main training loop
    # global_step = 0
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Ensure DistributedSampler is reset at the beginning of each epoch
        train_sampler.set_epoch(epoch)
        
        print(f"[Rank {rank}] Starting Epoch {epoch}")
        
        # Set training mode
        unet.train()
        cloth_encoder.train()
        
        total_loss = 0
        loss_count = 0
        
        progress_bar = tqdm(total=len(dataloader)) if rank == 0 else None
        if progress_bar is not None:
            progress_bar.set_description(f"Epoch {epoch}")
        
        # Whether to use LPIPS loss (warmup strategy)
        use_lpips = epoch >= LPIPS_WARMUP_EPOCHS
        
        # Counter for gradient accumulation
        accumulation_counter = 0
        total_loss_diffusion = 0.0
        total_loss_lpips = 0.0
        
        for step, batch in enumerate(dataloader):
            # Increment accumulation counter
            accumulation_counter += 1
            
            # Get input data
            clean_images = batch["pixel_values"].to(device)  # Target image
            pose = batch["pose"].to(device)                 # Pose condition
            agnostic_images = batch["agnostic"].to(device)  # Agnostic human image
            cloth_images = batch["cloth"].to(device)        # Clothing image

            # Use mixed precision training
            with torch.cuda.amp.autocast() if MIXED_PRECISION else nullcontext():
                # Get conditions from cloth encoder
                cloth_features = cloth_encoder(cloth_images)  # [B, D]
                
                # Prepare text prompt or use cloth condition
                if USE_TEXT_PROMPT and text_encoder is not None and tokenizer is not None:
                    # Use text prompt
                    prompts = ["a person wearing fashionable clothes"] * clean_images.shape[0]
                    inputs = tokenizer(
                        prompts, 
                        padding="max_length", 
                        max_length=tokenizer.model_max_length, 
                        return_tensors="pt"
                    ).to(device)
                    encoder_hidden_states = text_encoder(**inputs).last_hidden_state
                else:
                    # Don't use text prompt, use cloth features
                    # Expand cloth_features to shape expected by UNet [batch_size, seq_len, dim]
                    encoder_hidden_states = cloth_features.unsqueeze(1)
                
                # Encode target image to latent representation
                with torch.no_grad():
                    latents = vae.encode(clean_images).latent_dist.sample() * 0.18215
                    # Get latent representations for agnostic and pose
                    latent_agnostic = vae.encode(agnostic_images).latent_dist.sample() * 0.18215
                    latent_pose = vae.encode(pose).latent_dist.sample() * 0.18215
                
                # Random noise
                noise = torch.randn_like(latents)
                
                # Random timesteps - use different timesteps for each sample in the batch
                timesteps = torch.randint(
                    0, 
                    noise_scheduler.config.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=device
                ).long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Prepare concat input
                # 1. noisy_latent - [B, 4, H, W]
                # 2. latent_agnostic - [B, 4, H, W]
                # 3. latent_pose - [B, 4, H, W]
                
                # Concatenate all inputs - now a total of 12 channels(4+4+4)
                concat_input = torch.cat([
                    noisy_latents,        # Noisy image (zt)
                    latent_agnostic,      # Latent agnostic map E(xa)
                    latent_pose,          # Latent dense pose condition E(xp)
                ], dim=1)
                
                # Get noise prediction through UNet
                noise_pred = unet(
                    concat_input,
                    timesteps.float(),  # Ensure timestep is float type
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

                # Calculate diffusion loss
                loss_diffusion = torch.nn.functional.mse_loss(noise_pred, noise)
                loss = loss_diffusion
                total_loss_diffusion += loss_diffusion.item()
                
                # ============ LPIPS perceptual loss (applied to final generated image) ============
                if use_lpips:
                    with torch.no_grad():
                        # Calculate ᾱ_t (alpha_prod_t)
                        alpha_bar_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)  # shape [B, 1, 1, 1]

                        # Predict final latent representation (x0) using DDPM denoising formula
                        pred_latents = (noisy_latents - (1 - alpha_bar_t).sqrt() * noise_pred) / alpha_bar_t.sqrt()

                        # Decode to image
                        pred_images = vae.decode(pred_latents / 0.18215).sample  # shape: [B, 3, H, W]

                    # LPIPS perceptual loss (structure preservation)
                    lpips_loss = perceptual_loss_fn(pred_images, clean_images).mean()

                    # Set LPIPS loss weight strategy (can be fine-tuned)
                    lpips_weight = min(0.5, 0.1 * (epoch - LPIPS_WARMUP_EPOCHS + 1))
                    
                    # Add to total loss
                    loss = loss + lpips_weight * lpips_loss
                    total_loss_lpips += lpips_loss.item() * lpips_weight

                
                # Gradient accumulation: divide by accumulation steps
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Use mixed precision for backward propagation
            if MIXED_PRECISION:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Record loss
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS  # Restore actual loss value
            loss_count += 1
            
            # Only update parameters after completing gradient accumulation
            if accumulation_counter % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(dataloader) - 1:
                if MIXED_PRECISION:
                    # Clip gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(unet.parameters()) + 
                        list(cloth_encoder.parameters()),
                        max_norm=1.0
                    )
                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        list(unet.parameters()) + 
                        list(cloth_encoder.parameters()),
                        max_norm=1.0
                    )
                    # Update parameters
                    optimizer.step()
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Update learning rate
                lr_scheduler.step()
                
                # Reset accumulation counter
                accumulation_counter = 0
                
                # Update global step
                global_step += 1
            
            # Update progress bar
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item() * GRADIENT_ACCUMULATION_STEPS, lr=lr_scheduler.get_last_lr()[0])
            
            # Record training info to TensorBoard
            if rank == 0 and global_step % 10 == 0:
                writer.add_scalar("train/loss", loss.item() * GRADIENT_ACCUMULATION_STEPS, global_step)
                writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/loss_diffusion", total_loss_diffusion / GRADIENT_ACCUMULATION_STEPS, global_step)
                if use_lpips:
                    writer.add_scalar("train/loss_lpips", total_loss_lpips / accumulation_counter, global_step)
        
        if progress_bar is not None:
            progress_bar.close()
        
        # Calculate average loss
        total_loss_tensor = torch.tensor([total_loss], device=device)
        loss_count_tensor = torch.tensor([loss_count], device=device)
        
        # Collect and average losses across all processes
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count_tensor, op=dist.ReduceOp.SUM)
        
        avg_loss = total_loss_tensor.item() / loss_count_tensor.item()
        
        if rank == 0:
            print(f"Epoch {epoch}: Average loss = {avg_loss:.4f}")
            writer.add_scalar("train/epoch_loss", avg_loss, epoch)
        
        # Ensure all processes complete this epoch's training
        dist.barrier()
        
        # Save model and perform evaluation
        if rank == 0 and ((epoch >= NUM_EPOCHS // 2 and epoch % 40 == 0) or epoch == NUM_EPOCHS - 1):
            # Create save directory
            epoch_dir = os.path.join(OUTPUT_DIR, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Save UNet model
            os.makedirs(os.path.join(epoch_dir, "unet"), exist_ok=True)
            torch.save(unet.module.state_dict(), os.path.join(epoch_dir, "unet", "diffusion_pytorch_model.bin"))
            # Also save configuration file for later loading
            # unet.module.config.save_pretrained(os.path.join(epoch_dir, "unet"))
            # Save configuration file, but don't use save_pretrained method
            config_dict = dict(unet.module.config)
            config_dict["_class_name"] = unet.module.__class__.__name__
            config_dict["_diffusers_version"] = "0.16.1"  # Use a reasonable version number
            
            # Save configuration as JSON
            import json
            with open(os.path.join(epoch_dir, "unet", "config.json"), "w") as f:
                json.dump(config_dict, f)
            
            # Save cloth encoder
            torch.save(cloth_encoder.module.state_dict(), os.path.join(epoch_dir, "cloth_encoder.pth"))

            # Save training metadata for resuming
            training_info = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": lr_scheduler.get_last_lr()[0],
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "seed": SEED,
                "mixed_precision": MIXED_PRECISION,
            }
                
            with open(os.path.join(epoch_dir, "training_info.json"), "w") as f:
                json.dump(training_info, f)
            
            print(f"[Rank {rank}] Saved checkpoint at epoch {epoch}")
            
            # Perform evaluation
            if SAVE_EVAL:
                # Perform evaluation with current model
                evaluate(
                    rank=rank,
                    unet=unet.module,  # Use unwrapped model
                    cloth_encoder=cloth_encoder.module,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    val_loader=val_loader,
                    epoch=epoch,
                    output_dir=OUTPUT_DIR,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer
                )
            
            print(f"[Rank {rank}] Model saving completed")
        
        # Ensure all processes complete this epoch's processing
        dist.barrier()
    
    # Training finished, clean up
    if rank == 0:
        writer.close()
        print("Training finished!")
    
    # Clean up distributed environment
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VITON direct concatenation training")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="GPU IDs to use, comma separated")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Total number of epochs to train")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Base learning rate")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save models and logs")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--mixed_precision", action="store_true", help="Whether to use mixed precision training")
    parser.add_argument("--clip_model", type=str, default=CLIP_MODEL_NAME, help="CLIP model name")
    parser.add_argument("--resume_from", type=str, default=None, 
                    help="Resume training from a specific checkpoint directory (e.g., 'epoch_100')")
    
    args = parser.parse_args()
    
    # Update global configuration
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.lr
    OUTPUT_DIR = args.output_dir
    SEED = args.seed
    MIXED_PRECISION = args.mixed_precision
    CLIP_MODEL_NAME = args.clip_model
    
    # Get number of available GPUs
    gpu_ids = [int(id) for id in args.gpu_ids.split(",")]
    world_size = len(gpu_ids)
    
    # Check and create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set up signal handling for cleaning up processes when training is interrupted
    def signal_handler(sig, frame):
        print("Training interrupted, cleaning up processes...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Launch with multiprocessing
    print(f"Using {world_size} GPUs for distributed training: {gpu_ids}")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    # Launch multi-process training
    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )