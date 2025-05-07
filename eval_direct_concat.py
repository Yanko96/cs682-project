import os
import torch
import torch.nn as nn
import argparse
import lpips
import imageio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
    PNDMScheduler,  # Added for PLMS sampling
)
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPImageProcessor

# New imports for FID and SSIM
from pytorch_fid import fid_score
from skimage.metrics import structural_similarity as ssim
import cv2
import tempfile
import shutil
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.models import inception_v3
from scipy import linalg


# Default parameters
DATA_ROOT = "/PATH/TO/VITON/VAL/DATASET"
CHECKPOINT_DIR = "/PATH/TO/CHECKPOINT/DIR"
PRETRAINED_MODEL = "runwayml/stable-diffusion-v1-5"
BATCH_SIZE = 4
IMAGE_SIZE = (512, 384)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
USE_TEXT_PROMPT = False
EVAL_NUM_SAMPLES = 16


class VITONDirectConcatDataset(Dataset):
    def __init__(self, root, use_cloth_agnostic=True):
        self.pose_dir = os.path.join(root, "pose")
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
        
        # Convert from [0,1] to [-1,1] range
        pose = pose * 2 - 1  # Dense pose condition
        cloth = cloth * 2 - 1  
        person = person * 2 - 1
        agnostic = agnostic * 2 - 1  # Agnostic map
        
        return {
            "name": name, 
            "pixel_values": person,  # Target image
            "pose": pose,           # Dense pose condition
            "cloth": cloth,         # Garment image for conditioning
            "agnostic": agnostic,   # Agnostic map
            "parse": parse,         # Parsing map
        }


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


def load_and_modify_unet(pretrained_model_path, in_channels=12, device=None):
    """Load UNet and modify first convolution layer to accept in_channels input channels"""
    # Load original UNet
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    if device is not None:
        unet = unet.to(device)
    
    # Save original convolution layer weights
    orig_conv_weight = unet.conv_in.weight.data.clone()
    orig_conv_bias = None
    if unet.conv_in.bias is not None:
        orig_conv_bias = unet.conv_in.bias.data.clone()
    
    # Create new first convolution layer with more input channels
    new_conv = nn.Conv2d(
        in_channels=in_channels,  # New input channels
        out_channels=unet.conv_in.out_channels,
        kernel_size=unet.conv_in.kernel_size,
        stride=unet.conv_in.stride,
        padding=unet.conv_in.padding,
        bias=unet.conv_in.bias is not None
    ).to(device)
    
    # Copy original weights to first 4 channels
    with torch.no_grad():
        new_conv.weight.data[:, :4, :, :] = orig_conv_weight
        new_conv.weight.data[:, 4:, :, :] = 0  # Initialize remaining channels to 0
        if orig_conv_bias is not None:
            new_conv.bias.data = orig_conv_bias
    
    # Replace first convolution layer
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
    guidance_scale=7.5,
    use_plms=False
):
    """Generate an image using the concat input method"""
    if device is None:
        device = agnostic_image.device
    
    # Prepare inputs
    batch_size = agnostic_image.shape[0]
    
    # Get latent space shape
    height, width = agnostic_image.shape[-2:]
    latent_height = height // 8
    latent_width = width // 8
    
    # Generate pure noise for each sample (standard normal distribution)
    latent = torch.randn(batch_size, 4, latent_height, latent_width, device=device)
    
    # Get cloth features
    cloth_features = cloth_encoder(cloth_image)
    
    # Prepare text condition or cloth condition
    if USE_TEXT_PROMPT and text_encoder is not None and tokenizer is not None:
        text_input = tokenizer(
            [prompt] * batch_size, 
            padding="max_length", 
            max_length=tokenizer.model_max_length, 
            return_tensors="pt"
        ).to(device)
        encoder_hidden_states = text_encoder(**text_input).last_hidden_state
    else:
        # Expand cloth features to shape expected by UNet [batch_size, seq_len, dim]
        # Assuming UNet expects 77 tokens with dimension 768
        encoder_hidden_states = cloth_features.unsqueeze(1).expand(-1, 77, -1)
    
    # Prepare diffusion process
    scheduler = noise_scheduler
    # Explicitly set timesteps and move to correct device
    timesteps = scheduler.timesteps.to(device)
    
    # Start with random noise
    noisy_latent = latent
    
    # Get latent space agnostic and pose
    with torch.no_grad():
        latent_agnostic = vae.encode(agnostic_image).latent_dist.sample() * 0.18215
        latent_pose = vae.encode(pose).latent_dist.sample() * 0.18215
    
    # Create unconditional embeddings (for classifier-free guidance)
    do_classifier_free_guidance = guidance_scale > 1.0
    if do_classifier_free_guidance:
        if USE_TEXT_PROMPT and text_encoder is not None and tokenizer is not None:
            # Create unconditional embeddings with text encoder
            uncond_input = tokenizer(
                [""] * batch_size, 
                padding="max_length", 
                max_length=tokenizer.model_max_length, 
                return_tensors="pt"
            ).to(device)
            uncond_embeddings = text_encoder(**uncond_input).last_hidden_state
        else:
            # When not using text prompts, create all-zero embeddings as unconditional input
            uncond_embeddings = torch.zeros_like(encoder_hidden_states)
        
        # Concatenate conditional and unconditional embeddings
        encoder_hidden_states = torch.cat([uncond_embeddings, encoder_hidden_states])
    
    # Denoising loop
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
            
            # Concatenate all inputs - total of 12 channels (4+4+4)
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
        
        # Get noise prediction from UNet
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
        
        # Denoising step
        noisy_latent = scheduler.step(noise_pred, t, noisy_latent).prev_sample
    
    # Decode final latent representation
    with torch.no_grad():
        image = vae.decode(noisy_latent / 0.18215).sample
    
    # Convert to image space [-1, 1] -> [0, 1]
    image = (image + 1) / 2
    image = image.clamp(0, 1)
    
    return image


# Initialize the Inception model for FID calculation
def get_inception_model(device):
    """Initialize Inception V3 model for feature extraction (needed for FID)"""
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()  # Remove classification layer
    inception_model.eval()
    return inception_model.to(device)


# Calculate features for FID
def get_inception_features(img_tensor, inception_model):
    """Extract features from inception model for a single image"""
    # Ensure img_tensor is in correct format: [1, 3, 299, 299]
    if img_tensor.shape[-1] != 299 or img_tensor.shape[-2] != 299:
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    
    with torch.no_grad():
        features = inception_model(img_tensor)
    return features


# Calculate FID for a single image pair
def calculate_single_fid(gen_img_tensor, target_img_tensor, inception_model):
    """
    Calculate an approximation of FID for a single image pair
    
    Args:
        gen_img_tensor: Generated image tensor [1, 3, H, W]
        target_img_tensor: Target image tensor [1, 3, H, W]
        inception_model: Pre-trained Inception v3 model
        
    Returns:
        FID approximation
    """
    # Extract features
    gen_features = get_inception_features(gen_img_tensor, inception_model)
    target_features = get_inception_features(target_img_tensor, inception_model)
    
    # Calculate mean and covariance for single samples
    gen_mu = gen_features.mean(0, keepdim=True)
    target_mu = target_features.mean(0, keepdim=True)
    
    # Use L2 distance as a simple FID approximation for single images
    fid_approx = torch.norm(gen_mu - target_mu).item()
    return fid_approx


# Calculate SSIM for a single image pair
def calculate_ssim(gen_img, target_img):
    """
    Calculate SSIM between a generated and target image.
    
    Args:
        gen_img: Generated image as PIL Image, tensor, or numpy array
        target_img: Target image as PIL Image, tensor, or numpy array
        
    Returns:
        SSIM score
    """
    # Convert to numpy arrays if they're tensors or PIL images
    if isinstance(gen_img, torch.Tensor):
        gen_img = to_pil_image(gen_img)
    if isinstance(target_img, torch.Tensor):
        target_img = to_pil_image(target_img)
        
    # Convert PIL images to numpy arrays
    if isinstance(gen_img, Image.Image):
        gen_img = np.array(gen_img)
    if isinstance(target_img, Image.Image):
        target_img = np.array(target_img)
        
    # Convert to grayscale if needed for SSIM
    if gen_img.ndim == 3 and gen_img.shape[2] == 3:
        gen_img_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
        target_img_gray = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
    else:
        gen_img_gray = gen_img
        target_img_gray = target_img
        
    # Calculate SSIM
    ssim_value = ssim(gen_img_gray, target_img_gray, data_range=255)
    return ssim_value


# Calculate standard FID score for two directories
def calculate_fid(generated_dir, target_dir):
    """
    Calculate FID score between directories of generated and target images.
    
    Args:
        generated_dir: Directory containing generated images
        target_dir: Directory containing target/ground truth images
        
    Returns:
        FID score
    """
    try:
        # Use pytorch_fid library to calculate FID
        fid_value = fid_score.calculate_fid_given_paths(
            [generated_dir, target_dir],
            batch_size=50,
            device='cuda',
            dims=2048
        )
        return fid_value
    except Exception as e:
        print(f"Error calculating FID score: {e}")
        return float('nan')


def main():
    parser = argparse.ArgumentParser(description="VITON Direct Concat Evaluation")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Path to validation dataset")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR, help="Path to checkpoint directory")
    parser.add_argument("--epoch", type=int, default=-1, help="Specific epoch to evaluate (-1 for latest)")
    parser.add_argument("--pretrained_model", type=str, default=PRETRAINED_MODEL, help="Pretrained diffusion model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=EVAL_NUM_SAMPLES, help="Number of samples to evaluate")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=100, help="Number of inference steps")
    parser.add_argument("--use_text_prompt", action="store_true", help="Use text prompts instead of cloth features")
    parser.add_argument("--use_plms", action="store_true", help="Use PLMS sampler for faster inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Set USE_TEXT_PROMPT global variable
    global USE_TEXT_PROMPT
    USE_TEXT_PROMPT = args.use_text_prompt
    
    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    val_dataset = VITONDirectConcatDataset(args.data_root, use_cloth_agnostic=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Find checkpoint to evaluate
    if args.epoch >= 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{args.epoch}")
    else:
        # Find the latest epoch
        epoch_dirs = [d for d in os.listdir(args.checkpoint_dir) if d.startswith("epoch_")]
        epoch_nums = [int(d.split("_")[1]) for d in epoch_dirs]
        latest_epoch = max(epoch_nums) if epoch_nums else None
        
        if latest_epoch is None:
            raise ValueError(f"No epoch checkpoints found in {args.checkpoint_dir}")
        
        checkpoint_path = os.path.join(args.checkpoint_dir, f"epoch_{latest_epoch}")
        print(f"Using latest checkpoint: epoch_{latest_epoch}")
    
    # Setup output directory
    if args.output_dir is None:
        output_dir = os.path.join(checkpoint_path, "evaluation")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAE from pretrained model
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    
    # Load noise scheduler
    print("Loading noise scheduler...")
    if args.use_plms:
        print("Using PLMS sampler for inference")
        noise_scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model, 
            subfolder="scheduler",
            algorithm_type="pndm"  # PLMS is an option in the PNDM scheduler
        )
        # Configure for PLMS
        noise_scheduler.set_timesteps(args.num_inference_steps)
    else:
        print("Using DDPM sampler for inference")
        noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")
        # Set timesteps manually
        noise_scheduler.set_timesteps(args.num_inference_steps)
    
    # Load text encoder and tokenizer if using text prompts
    if USE_TEXT_PROMPT:
        print("Loading text encoder and tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder")
        text_encoder = text_encoder.to(device)
        text_encoder.eval()
    else:
        tokenizer = None
        text_encoder = None
    
    # Load UNet from checkpoint
    print("Loading UNet from checkpoint...")
    unet_path = os.path.join(checkpoint_path, "unet", "diffusion_pytorch_model.bin")
    
    # First load the base model
    unet = load_and_modify_unet(args.pretrained_model, in_channels=12, device=device)
    
    # Then load checkpoint weights
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()
    
    # Load cloth encoder from checkpoint
    print("Loading cloth encoder from checkpoint...")
    cloth_encoder_path = os.path.join(checkpoint_path, "cloth_encoder.pth")
    
    # Get UNet cross-attention dimension (usually 768)
    unet_cross_attention_dim = unet.config.cross_attention_dim
    
    # Initialize cloth encoder
    cloth_encoder = SimplifiedClothEncoder(clip_model_name=CLIP_MODEL_NAME, output_dim=unet_cross_attention_dim)
    cloth_encoder.load_state_dict(torch.load(cloth_encoder_path, map_location=device))
    cloth_encoder = cloth_encoder.to(device)
    cloth_encoder.eval()
    
    # Initialize LPIPS for evaluation
    perceptual_loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Create output directories
    GEN_DIR = os.path.join(output_dir, "generated")
    GT_DIR = os.path.join(output_dir, "groundtruth")
    COMPARE_DIR = os.path.join(output_dir, "comparison")
    CONTROLS_DIR = os.path.join(output_dir, "controls")
    
    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    os.makedirs(COMPARE_DIR, exist_ok=True)
    os.makedirs(CONTROLS_DIR, exist_ok=True)
    
    # Run evaluation
    print("Starting evaluation...")
    total_lpips = 0.0
    total_ssim = 0.0
    total_fid_approx = 0.0
    sample_count = 0
    gif_frames = []
    
    # Initialize Inception model for per-sample FID approximation
    inception_model = get_inception_model(device)
    
    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Evaluating")):
        # if sample_count >= args.num_samples:
        if batch_idx >= args.num_samples:
            break
        
        # Get validation batch
        names = batch["name"]
        poses = batch["pose"].to(device)
        targets = batch["pixel_values"].to(device)
        agnostics = batch["agnostic"].to(device)
        cloth_imgs = batch["cloth"].to(device)
        
        # Generate images
        images = []
        gen_tensors = []
        
        with torch.no_grad():
            for i in range(len(names)):
                # if sample_count >= args.num_samples:
                #     break
                    
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
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    use_plms=args.use_plms
                )
                
                # Save generated results
                gen_tensor = gen_img[0].cpu()
                gen_tensors.append(gen_tensor)
                img_pil = transforms.ToPILImage()(gen_tensor)
                images.append(img_pil)
                
                # Save individual sample
                name = names[i]
                
                # Save generated image
                img_pil.save(os.path.join(GEN_DIR, f"{name}.png"))
                
                # Save control inputs
                save_image((poses[i] + 1) / 2, os.path.join(CONTROLS_DIR, f"{name}_pose.png"))
                save_image((agnostics[i] + 1) / 2, os.path.join(CONTROLS_DIR, f"{name}_agnostic.png"))
                save_image((cloth_imgs[i] + 1) / 2, os.path.join(CONTROLS_DIR, f"{name}_cloth.png"))
                
                # Save ground truth image
                target_image = ((targets[i] + 1) / 2.0).clamp(0, 1)
                save_image(target_image, os.path.join(GT_DIR, f"{name}.png"))
                
                # Convert images to tensors for metrics calculation
                pred_tensor = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
                # Convert to [-1,1] range to match targets
                pred_tensor_norm = pred_tensor * 2 - 1
                target_tensor = targets[i].unsqueeze(0)
                
                # 1. Calculate LPIPS
                lpips_value = perceptual_loss_fn(pred_tensor_norm, target_tensor).item()
                total_lpips += lpips_value
                
                # 2. Calculate SSIM
                ssim_value = calculate_ssim(img_pil, to_pil_image(target_image.cpu()))
                total_ssim += ssim_value
                
                # 3. Calculate per-sample FID approximation
                fid_approx = calculate_single_fid(pred_tensor, target_image.unsqueeze(0).to(device), inception_model)
                total_fid_approx += fid_approx
                
                # Generate comparison: control inputs | generated image | ground truth
                pose_tensor = ((poses[i] + 1) / 2.0).cpu()
                cloth_tensor = ((cloth_imgs[i] + 1) / 2.0).cpu()
                agnostic_tensor = ((agnostics[i] + 1) / 2.0).cpu()
                gen_tensor = gen_tensors[-1]
                target_tensor_cpu = target_image.cpu()
                
                # Create comparison grid
                comparison = torch.cat([
                    torch.cat([pose_tensor, cloth_tensor], dim=2),
                    torch.cat([agnostic_tensor, gen_tensor], dim=2),
                    torch.cat([gen_tensor, target_tensor_cpu], dim=2)
                ], dim=1)
                
                save_image(comparison, os.path.join(COMPARE_DIR, f"{name}_comparison.png"))
                gif_frames.append(np.array(transforms.ToPILImage()(comparison)) * 255)
                
                # Print all metrics for this sample
                sample_count += 1
                print(f"Generated sample {sample_count}/{args.num_samples}, LPIPS: {lpips_value:.4f}, SSIM: {ssim_value:.4f}, FID approx: {fid_approx:.4f}")
    
    # Save GIF and evaluation metrics
    if gif_frames:
        imageio.mimsave(os.path.join(output_dir, "evaluation.gif"), [frame.astype(np.uint8) for frame in gif_frames], duration=0.5)
    
    # Calculate FID score on the directories (full dataset FID)
    print("Calculating full dataset FID score...")
    fid_value = calculate_fid(GEN_DIR, GT_DIR)
    print(f"Full dataset FID Score: {fid_value:.4f}")
    
    # Calculate average metrics
    if sample_count > 0:
        avg_lpips = total_lpips / sample_count
        avg_ssim = total_ssim / sample_count
        avg_fid_approx = total_fid_approx / sample_count
        
        # Save metrics to file
        with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
            f.write(f"Number of samples: {sample_count}\n")
            f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average per-sample FID approximation: {avg_fid_approx:.4f}\n")
            f.write(f"Full dataset FID Score: {fid_value:.4f}\n")
        
        print(f"Evaluation complete.")
        print(f"Average LPIPS: {avg_lpips:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average per-sample FID approximation: {avg_fid_approx:.4f}")
        print(f"Full dataset FID Score: {fid_value:.4f}")
    else:
        print("No samples were evaluated.")


if __name__ == "__main__":
    main()