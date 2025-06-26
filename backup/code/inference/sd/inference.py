import base64
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL


class HfModelId(str, Enum):
    SD_V1_5: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    SD_VAE: str = "stabilityai/sd-vae-ft-mse"


def model_fn(model_dir: str) -> Any:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = StableDiffusionPipeline.from_pretrained(
        HfModelId.SD_V1_5.value, torch_dtype=torch.float16
    ).to(device)

    model.vae = AutoencoderKL.from_pretrained(
        HfModelId.SD_VAE.value, torch_dtype=torch.float16
    ).to(device)

    model.scheduler = EulerDiscreteScheduler.from_config(
        model.scheduler.config, use_karras_sigmas=True
    )

    model.load_lora_weights(model_dir)

    return model


def predict_fn(data: Dict[str, Any], model: Any) -> Dict[str, List[str]]:
    prompt = data.pop("prompt", "")
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", 42)
    cross_attention_scale = data.pop("cross_attention_scale", 0.5)

    negative_prompt = negative_prompt if len(negative_prompt) > 0 else None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = torch.Generator(device=device).manual_seed(seed)
    generated_images = model(
        prompt,
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
