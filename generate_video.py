#!/usr/bin/env python3
"""Generate a video from a natural language prompt using local CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video


DEFAULT_MODEL = "damo-vilab/text-to-video-ms-1.7b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="Text prompt to generate a video.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Hugging Face model ID or local path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.mp4"),
        help="Output video path (default: output.mp4).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of frames to generate (default: 16).",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of diffusion steps (default: 50).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=9.0,
        help="Classifier-free guidance scale (default: 9.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. This script requires local CUDA.")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe = pipe.to("cuda")

    result = pipe(
        prompt=args.prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames, args.output.as_posix())
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
