from diffusers import StableVideoDiffusionPipeline
import torch

# Load the pre-trained Stable Video Diffusion model
model_id = "stabilityai/stable-video-diffusion-img2vid"
pipe = StableVideoDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Move model to GPU for faster inference

# Load an image (replace with your own)
from PIL import Image
image = Image.open("your_image.png").convert("RGB")

# Generate a video from the image
video_frames = pipe(image, num_inference_steps=25).frames

# Save the generated video
import imageio
imageio.mimsave("generated_video.mp4", video_frames, fps=10)