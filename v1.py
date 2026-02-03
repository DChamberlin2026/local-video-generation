#!/usr/bin/env python3
"""
text2video_local.py

Generate a short video clip from a natural language text prompt using:
1) Stable Diffusion (text -> image)
2) Stable Video Diffusion (image -> video)

- Runs locally on CUDA (if available).
- Requires: torch, diffusers, transformers, accelerate, safetensors, pillow, imageio
- Outputs: 
    * frames/XXXX/frame_000.png ... frame_N.png
    * out.gif (animated video)

Usage:
    python text2video_local.py --prompt "a cute robot walking in a neon city at night"

You can tweak options like:
    --num_frames 25 --fps 8 --height 576 --width 1024
"""

import argparse
import os
from datetime import datetime

import torch
import numpy as np
from PIL import Image
import imageio

from diffusers import (
    StableDiffusionPipeline,
    StableVideoDiffusionPipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local text-to-video with Stable Diffusion + Stable Video Diffusion")
    parser.add_argument(
        "--prompt",
        type=str,
        required=False,
        default=None,
        help="Text prompt describing the scene you want in the video.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, blurry, artifact, distorted, text, watermark",
        help="Negative prompt to avoid unwanted artifacts.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
        help="Video frame height. SVD is trained for 576x1024; keep aspect similar.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Video frame width. SVD is trained for 576x1024; keep aspect similar.",
    )
    parser.add_argument(
        "--num_inference_steps_img",
        type=int,
        default=30,
        help="Diffusion steps for the initial image (higher = slower, better).",
    )
    parser.add_argument(
        "--guidance_scale_img",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale for image generation.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
        help="Number of frames for the video (SVD-XT is trained for 25).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Frames per second for the output GIF.",
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=40,
        help="Controls motion amount in SVD (0-255; higher = more motion).",
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help="Noise augmentation strength; higher = more deviation from input image.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output",
        help="Base directory to save frames and GIF.",
    )
    return parser.parse_args()


def ensure_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("[!] CUDA not found. Falling back to CPU (this will be slow).")
        return torch.device("cpu")


def load_image_pipe(device: torch.device) -> StableDiffusionPipeline:
    """
    Load a Stable Diffusion text-to-image pipeline for the first frame.
    You can swap 'runwayml/stable-diffusion-v1-5' for another SD model you prefer.
    """
    print("[*] Loading Stable Diffusion text-to-image pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing("max")
    return pipe


def load_video_pipe(device: torch.device) -> StableVideoDiffusionPipeline:
    """
    Load Stable Video Diffusion (image-to-video).
    Model: stabilityai/stable-video-diffusion-img2vid-xt
    """
    print("[*] Loading Stable Video Diffusion pipeline...")
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        variant="fp16" if device.type == "cuda" else None,
    )

    if device.type == "cuda":
        # Offload parts of the model between CPU/GPU to reduce VRAM usage
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe


def generate_initial_image(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    num_steps: int,
    guidance_scale: float,
    generator: torch.Generator,
) -> Image.Image:
    print("[*] Generating initial frame from text prompt...")
    img = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return img


def generate_video_frames(
    pipe: StableVideoDiffusionPipeline,
    init_image: Image.Image,
    num_frames: int,
    fps: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
    generator: torch.Generator,
):
    print("[*] Generating video frames with Stable Video Diffusion...")
    # SVD expects a PIL image of size 1024x576 (W x H). We already matched that via args,
    # but resize explicitly to be safe.
    init_image = init_image.resize((1024, 576), resample=Image.LANCZOS)

    result = pipe(
        init_image,
        num_frames=num_frames,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        generator=generator,
        decode_chunk_size=8,
    )

    # result.frames is a list [video_batch], where video_batch has shape (num_frames, H, W, 3)
    frames = result.frames[0]  # shape: (T, H, W, 3)
    print(f"[*] Generated {len(frames)} frames.")
    return frames


def save_frames_and_gif(frames, out_dir: str, fps: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(out_dir, f"frames_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)

    # Save individual frames as PNG
    pil_frames = []
    for idx, frame in enumerate(frames):
        # frame is a numpy array (H, W, 3) in [0,255] uint8
        if not isinstance(frame, Image.Image):
            img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        else:
            img = frame
        frame_path = os.path.join(frames_dir, f"frame_{idx:03d}.png")
        img.save(frame_path)
        pil_frames.append(img)

    print(f"[*] Saved {len(pil_frames)} frames to {frames_dir}")

    # Save animated GIF
    gif_path = os.path.join(out_dir, f"out_{timestamp}.gif")
    imageio.mimsave(gif_path, [np.array(f) for f in pil_frames], format="GIF", fps=fps)
    print(f"[*] Saved animated GIF to {gif_path}")


def main():
    args = parse_args()

    if args.prompt is None:
        # Fallback to interactive input if not supplied via CLI
        args.prompt = input("Enter your text prompt: ").strip()
        if not args.prompt:
            print("[!] Empty prompt, nothing to do.")
            return

    device = ensure_device()
    os.makedirs(args.out_dir, exist_ok=True)

    # For deterministic-ish output across runs with same settings
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    # 1) Load pipelines
    img_pipe = load_image_pipe(device)
    vid_pipe = load_video_pipe(device)

    # 2) Generate initial image
    init_image = generate_initial_image(
        img_pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_steps=args.num_inference_steps_img,
        guidance_scale=args.guidance_scale_img,
        generator=generator,
    )

    # 3) Generate video frames from the image
    frames = generate_video_frames(
        vid_pipe,
        init_image=init_image,
        num_frames=args.num_frames,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        generator=generator,
    )

    # 4) Save frames and GIF
    save_frames_and_gif(frames, out_dir=args.out_dir, fps=args.fps)

    print("[*] Done.")


if __name__ == "__main__":
    main()
