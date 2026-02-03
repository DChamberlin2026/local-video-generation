# local video generation
python tools for running text to video on local hardware

## Text-to-video script

`generate_video.py` runs a local CUDA text-to-video model using Diffusers. Example:

```bash
python generate_video.py "a cinematic sunset over the ocean" --output outputs/sunset.mp4
```

python text2video_local.py --prompt "a fluffy white dragon sleeping on a pile of glowing crystals, cinematic lighting"