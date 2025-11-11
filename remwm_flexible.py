# Final corrected script with all imports and batch processing
import sys
import click
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
# --- THIS IS THE CORRECTED LINE ---
from transformers import AutoProcessor, AutoModelForCausalLM
from iopaint.model_manager import ModelManager
from iopaint.schema import HDStrategy, LDMSampler, InpaintRequest as Config
import torch
import tqdm
from loguru import logger
from enum import Enum
import os
import tempfile
import shutil
import subprocess

try:
    from cv2.typing import MatLike
except ImportError:
    MatLike = np.ndarray

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"

def identify(task_prompt: TaskType, images: list, text_input: str, model: AutoModelForCausalLM, processor: AutoProcessor, device: str):
    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024, early_stopping=False, do_sample=False, num_beams=3)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
    
    results = []
    for i, generated_text in enumerate(generated_texts):
        image_size = (images[i].width, images[i].height)
        results.append(processor.post_process_generation(generated_text, task=task_prompt.value, image_size=image_size))
    return results

def get_watermark_masks_batch(images: list, model: AutoModelForCausalLM, processor: AutoProcessor, device: str, max_bbox_percent: float):
    text_input = "watermark Sora logo"
    parsed_answers = identify(TaskType.OPEN_VOCAB_DETECTION, images, text_input, model, processor, device)
    masks = []
    for i, parsed_answer in enumerate(parsed_answers):
        image = images[i]
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        detection_key = "<OPEN_VOCABULARY_DETECTION>"
        if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
            image_area = image.width * image.height
            for bbox in parsed_answer[detection_key]["bboxes"]:
                x1, y1, x2, y2 = map(int, bbox)
                bbox_area = (x2 - x1) * (y2 - y1)
                if (bbox_area / image_area) * 100 <= max_bbox_percent:
                    draw.rectangle([x1, y1, x2, y2], fill=255)
        masks.append(mask)
    return masks

def process_batch(images: list, masks: list, model_manager: ModelManager):
    config = Config(ldm_steps=50, ldm_sampler=LDMSampler.ddim, hd_strategy=HDStrategy.CROP)
    np_images = [np.array(img) for img in images]
    np_masks = [np.array(mask) for mask in masks]
    results = [model_manager(image, mask, config) for image, mask in zip(np_images, np_masks)]
    pil_results = []
    for result in results:
        if result.dtype in [np.float64, np.float32]:
            result = np.clip(result, 0, 255).astype(np.uint8)
        pil_results.append(Image.fromarray(result))
    return pil_results

def is_video_file(file_path):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return Path(file_path).suffix.lower() in video_extensions

def process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, max_bbox_percent, force_format, batch_size=8):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Error opening video file: {input_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_format = force_format.upper() if force_format else "MP4"
    output_file = Path(output_path).with_suffix(f".{output_format.lower()}")
    temp_dir = tempfile.mkdtemp()
    temp_video_path = Path(temp_dir) / f"temp_no_audio.{output_format.lower()}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(temp_video_path), fourcc, fps, (width, height))
    frame_batch = []
    with tqdm.tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frame_batch.append(pil_image)
            if len(frame_batch) == batch_size or (not ret and len(frame_batch) > 0):
                mask_batch = get_watermark_masks_batch(frame_batch, florence_model, florence_processor, device, max_bbox_percent)
                result_batch = process_batch(frame_batch, mask_batch, model_manager)
                for result_image in result_batch:
                    frame_result = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
                    out.write(frame_result)
                pbar.update(len(frame_batch))
                frame_batch = []
                torch.cuda.empty_cache()
            if not ret:
                break
    cap.release()
    out.release()
    try:
        logger.info("Merging video with original audio...")
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(temp_video_path), "-i", str(input_path), "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-shortest", str(output_file)]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        logger.error(f"FFmpeg error, saving video without audio: {e}")
        shutil.copy(str(temp_video_path), str(output_file))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    logger.info(f"Processing complete. Output saved to: {output_file}")
    return output_file

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option("--model", type=click.Choice(['lama', 'migan']), default='lama', help="Inpainting model to use.")
@click.option("--max-bbox-percent", default=10.0, help="Maximum percentage of the image that a bounding box can cover.")
@click.option("--force-format", type=click.Choice(["MP4", "AVI"], case_sensitive=False), default=None)
@click.option("--batch-size", default=8, type=int, help="Number of frames to process in a batch.")
def main(input_path: str, output_path: str, model: str, max_bbox_percent: float, force_format: str, batch_size: int):
    input_path = Path(input_path)
    output_path = Path(output_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    florence_model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True).to(device).eval()
    florence_processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)
    logger.info("Florence-2 Model loaded")
    model_manager = ModelManager(name=model, device=device)
    logger.info(f"{model.capitalize()} model loaded")
    if is_video_file(input_path):
        process_video(input_path, output_path, florence_model, florence_processor, model_manager, device, max_bbox_percent, force_format, batch_size)
    else:
        logger.error("This script is configured for video processing only.")
        sys.exit(1)

if __name__ == "__main__":
    main()
