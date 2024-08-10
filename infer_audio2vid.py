#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：EchoMimic
@File    ：audio2vid.py
@Author  ：juzhen.czy
@Date    ：2024/3/4 17:43 
"""
import argparse
import os

import random
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN

ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path is None and platform.system() in ["Linux", "Darwin"]:
    try:
        result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print(
                "FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH."
            )
            print("For example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv("PATH"):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/prompts/animation.yaml"
    )
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--facemusk_dilation_ratio", type=float, default=0.1)
    parser.add_argument("--facecrop_dilation_ratio", type=float, default=0.5)

    parser.add_argument("--context_frames", type=int, default=12)
    parser.add_argument("--context_overlap", type=int, default=3)

    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    return args


def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(
        filtered_bboxes, key=lambda x: (x[3] - x[1]) * (x[2] - x[0]), reverse=True
    )
    return sorted_bboxes[0]


def get_default_args():
    args = {
        "config": "./configs/prompts/animation.yaml",
        "W": 512,
        "H": 512,
        "L": 1200,
        "seed": 420,
        "facemusk_dilation_ratio": 0.1,
        "facecrop_dilation_ratio": 0.5,
        "context_frames": 12,
        "context_overlap": 3,
        "cfg": 2.5,
        "steps": 30,
        "sample_rate": 16000,
        "fps": 24,
        "device": "cuda",
    }
    return args


def main():
    # Default arguments
    args = {
        "config": "./configs/prompts/animation.yaml",
        "W": 512,
        "H": 512,
        "L": 1200,
        "seed": 420,
        "facemusk_dilation_ratio": 0.1,
        "facecrop_dilation_ratio": 0.5,
        "context_frames": 12,
        "context_overlap": 3,
        "cfg": 2.5,
        "steps": 30,
        "sample_rate": 16000,
        "fps": 24,
        "device": "cuda",
    }

    config = OmegaConf.load(args["config"])
    weight_dtype = torch.float16 if config.weight_dtype == "fp16" else torch.float32

    device = args["device"]
    if "cuda" in device and not torch.cuda.is_available():
        device = "cpu"

    infer_config = OmegaConf.load(config.inference_config)

    # Model initialization
    vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device=device)
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu")
    )

    if os.path.exists(config.motion_module_path):
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=infer_config.unet_additional_kwargs,
        ).to(dtype=weight_dtype, device=device)
    else:
        denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
            config.pretrained_base_model_path,
            "",
            subfolder="unet",
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
                "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim,
            },
        ).to(dtype=weight_dtype, device=device)
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )

    face_locator = FaceLocator(
        320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)
    ).to(dtype=weight_dtype, device="cuda")
    face_locator.load_state_dict(torch.load(config.face_locator_path))

    audio_processor = load_audio_model(
        model_path=config.audio_model_path, device=device
    )

    face_detector = MTCNN(
        image_size=320,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
    )

    # Pipeline setup
    scheduler = DDIMScheduler(
        **OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    )
    pipe = Audio2VideoPipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        face_locator=face_locator,
        scheduler=scheduler,
    ).to("cuda", dtype=weight_dtype)

    # Output directory setup
    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir = Path(
        f"output/{date_str}/{time_str}--seed_{args['seed']}-{args['W']}x{args['H']}"
    )
    save_dir.mkdir(exist_ok=True, parents=True)

    # Process each test case
    for ref_image_path, audio_paths in config["test_cases"].items():
        for audio_path in audio_paths:
            generator = torch.manual_seed(
                args["seed"] if args["seed"] > -1 else random.randint(100, 1000000)
            )

            ref_name = Path(ref_image_path).stem
            audio_name = Path(audio_path).stem

            # Face mask preparation
            face_img = cv2.imread(ref_image_path)
            face_mask = np.zeros(face_img.shape[:2], dtype="uint8")

            det_bboxes, probs = face_detector.detect(face_img)
            select_bbox = select_face(det_bboxes, probs)

            if select_bbox is None:
                face_mask.fill(255)
            else:
                xyxy = np.round(select_bbox[:4]).astype(int)
                rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
                r_pad = int((re - rb) * args["facemusk_dilation_ratio"])
                c_pad = int((ce - cb) * args["facemusk_dilation_ratio"])
                face_mask[rb - r_pad : re + r_pad, cb - c_pad : ce + c_pad] = 255

                # Face crop
                r_pad_crop = int((re - rb) * args["facecrop_dilation_ratio"])
                c_pad_crop = int((ce - cb) * args["facecrop_dilation_ratio"])
                crop_rect = [
                    max(0, cb - c_pad_crop),
                    max(0, rb - r_pad_crop),
                    min(ce + c_pad_crop, face_img.shape[1]),
                    min(re + c_pad_crop, face_img.shape[0]),
                ]
                face_img, _ = crop_and_pad(face_img, crop_rect)
                face_mask, _ = crop_and_pad(face_mask, crop_rect)

            face_img = cv2.resize(face_img, (args["W"], args["H"]))
            face_mask = cv2.resize(face_mask, (args["W"], args["H"]))

            ref_image_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_mask_tensor = (
                torch.tensor(face_mask, dtype=weight_dtype, device="cuda")
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                / 255.0
            )

            # Generate video
            video = pipe(
                ref_image_pil,
                audio_path,
                face_mask_tensor,
                args["W"],
                args["H"],
                args["L"],
                args["steps"],
                args["cfg"],
                generator=generator,
                audio_sample_rate=args["sample_rate"],
                context_frames=args["context_frames"],
                fps=args["fps"],
                context_overlap=args["context_overlap"],
            ).videos

            # Save video
            output_path = (
                save_dir
                / f"{ref_name}_{audio_name}_{args['H']}x{args['W']}_{int(args['cfg'])}_{time_str}.mp4"
            )
            save_videos_grid(video, str(output_path), n_rows=1, fps=args["fps"])

            # Add audio to video
            video_clip = VideoFileClip(str(output_path))
            audio_clip = AudioFileClip(audio_path)
            video_clip = video_clip.set_audio(audio_clip)
            output_path_with_audio = output_path.with_name(
                f"{output_path.stem}_withaudio.mp4"
            )
            video_clip.write_videofile(
                str(output_path_with_audio), codec="libx264", audio_codec="aac"
            )
            print(f"Video saved: {output_path_with_audio}")


if __name__ == "__main__":
    main()
