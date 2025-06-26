import base64
from enum import Enum
from io import BytesIO
from typing import Any, Dict, Final, List
import torch
from diffusers import DiffusionPipeline


class HfModelId(str, Enum):
    SDXL_V1_0_BASE: str = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_V1_0_REFINER: str = "stabilityai/stable-diffusion-xl-refiner-1.0"


ENABLE_MODEL_CPU_OFFLOAD: Final = True
USE_REFINER: Final = False


def model_fn(model_dir: str) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionPipeline.from_pretrained(
        HfModelId.SDXL_V1_0_BASE.value,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    _ = (
        model.enable_model_cpu_offload()
        if ENABLE_MODEL_CPU_OFFLOAD
        else model.to(device)
    )

    model.load_lora_weights(model_dir)

    if USE_REFINER:
        refiner = DiffusionPipeline.from_pretrained(
            HfModelId.SDXL_V1_0_REFINER.value,
            text_encoder_2=model.text_encoder_2,
            vae=model.vae,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        _ = (
            refiner.enable_model_cpu_offload()
            if ENABLE_MODEL_CPU_OFFLOAD
            else refiner.to(device)
        )
    else:
        refiner = None

    return {"model": model, "refiner": refiner}


def predict_fn(data: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, List[str]]:
    prompt = data.pop("prompt", "")
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", 42)
    high_noise_frac = data.pop("high_noise_frac", 0.7)
    cross_attention_scale = data.pop("cross_attention_scale", 0.5)

    negative_prompt = negative_prompt if len(negative_prompt) > 0 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, refiner = model["model"], model["refiner"]
    generator = torch.Generator(device=device).manual_seed(seed)

    if USE_REFINER:
        image = model(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            denoising_end=high_noise_frac,
            generator=generator,
            output_type="latent",
            cross_attention_kwargs={"scale": cross_attention_scale},
        )["images"]
        generated_images = refiner(
            prompt=prompt,
            image=image,
            num_inference_steps=num_inference_steps,
            denoising_start=high_noise_frac,
        )["images"]

    else:
        generated_images = model(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            cross_attention_kwargs={"scale": cross_attention_scale},
        )["images"]

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"generated_images": encoded_images, "prompt": prompt}
