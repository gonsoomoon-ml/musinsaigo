import argparse
import json
import os
import sys
from shutil import copyfile
from typing import Final, Optional
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm

sys.path.append("/tmp")
from utils.enums import HfModelId
from utils.logger import logger
from utils.misc import arg_as_bool

MODEL_GEN_CONFIG: Final = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "negative_prompt": "other people in the background",
    "seed": 42,
}


def get_loc(src_len: int, tgt_len: int) -> int:
    return round((src_len - tgt_len) // 2)


def outpaint_image(
    base_dir: str,
    dataset_prefix: str,
    images_prefix: str,
    hf_model_id: str,
    outpaint_prompt: str,
    resolution: int,
    run_compile: bool,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    raw_dataset_dir = os.path.join(base_dir, f"raw_{dataset_prefix}")
    proc_dataset_dir = os.path.join(base_dir, f"proc_{dataset_prefix}")
    models_dir = os.path.join(base_dir, "models")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.join(proc_dataset_dir, images_prefix), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StableDiffusionInpaintPipeline.from_pretrained(
        hf_model_id,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)

    if run_compile:
        model.unet.to(memory_format=torch.channels_last)
        torch.compile(model.unet, mode="reduce-overhead", fullgraph=True)

    raw_metadata_path = os.path.join(raw_dataset_dir, "metadata.jsonl")
    proc_metadata_path = os.path.join(proc_dataset_dir, "metadata.jsonl")

    generator = torch.Generator(device=device).manual_seed(seed)

    with open(raw_metadata_path, "r", encoding="utf-8") as input_file:
        for line in tqdm(input_file):
            metadata = json.loads(line)
            image_path, text = metadata["file_name"], metadata["text"]
            prompt = outpaint_prompt if len(outpaint_prompt) > 0 else text

            src_image = Image.open(os.path.join(raw_dataset_dir, image_path))
            src_image_size = np.array(src_image).shape

            max_len = max(src_image_size[0], src_image_size[1])
            tgt_image_size = max_len, max_len

            rgba_image = Image.new(mode="RGBA", size=tgt_image_size)
            Image.Image.paste(
                rgba_image,
                src_image,
                (
                    get_loc(tgt_image_size[1], src_image_size[1]),
                    get_loc(tgt_image_size[0], src_image_size[0]),
                ),
            )
            rgb_image = rgba_image.convert("RGB")

            full_mask = np.array(rgba_image)[:, :, 3] == 0
            full_mask = full_mask.astype(np.uint8) * 255
            full_mask = np.dstack([np.array(full_mask)] * 3)
            mask_image = Image.fromarray(full_mask)

            tgt_image = model(
                prompt=prompt,
                height=resolution,
                width=resolution,
                image=rgb_image.resize((resolution, resolution)),
                mask_image=mask_image.resize((resolution, resolution)),
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            ).images[0]

            tgt_image_size = round(
                resolution * src_image_size[0] / tgt_image_size[0]
            ), round(resolution * src_image_size[1] / tgt_image_size[1])

            Image.Image.paste(
                tgt_image,
                src_image.resize(
                    (
                        tgt_image_size[1],
                        tgt_image_size[0],
                    )
                ),
                (
                    get_loc(resolution, tgt_image_size[1]),
                    get_loc(resolution, tgt_image_size[0]),
                ),
            )

            tgt_image.save(os.path.join(proc_dataset_dir, image_path))

    copyfile(raw_metadata_path, proc_metadata_path)


if __name__ == "__main__":
    logger.info("The image outpainting task started...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/opt/ml/processing")
    parser.add_argument("--dataset-prefix", type=str, default="dataset")
    parser.add_argument("--images-prefix", type=str, default="images")
    parser.add_argument("--outpaint-prompt", type=str, default="")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--run-compile", type=arg_as_bool, default=False)

    args, _ = parser.parse_known_args()

    outpaint_image(
        args.base_dir,
        args.dataset_prefix,
        args.images_prefix,
        HfModelId.SD_INPAINT.value,
        args.outpaint_prompt,
        args.resolution,
        args.run_compile,
        **MODEL_GEN_CONFIG,
    )

    logger.info("The image outpainting task ended successfully.")
