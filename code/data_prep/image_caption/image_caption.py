import argparse
import glob
import json
import os
import sys
from typing import Final, Optional
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    InstructBlipForConditionalGeneration,
)
from tqdm import tqdm

sys.path.append("/tmp")
from utils.enums import HfModelId
from utils.logger import logger
from utils.misc import arg_as_bool, get_max_memory


MODEL_GEN_CONFIG: Final = {
    "max_length": 512,
    "min_length": 81,
    "do_sample": True,
    "num_beams": 5,
    "temperature": 1.0,
    "top_p": 0.9,
    "repetition_penalty": 1.5,
    "length_penalty": 1.0,
}


def caption_images(
    base_dir: str,
    images_prefix: str,
    captions_prefix: str,
    hf_model_id: str,
    caption_prompt: str,
    prompt_prefix: str,
    prompt_suffix: str,
    use_auto_device_map: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    length_penalty: Optional[float] = None,
) -> None:
    images_dir = os.path.join(base_dir, images_prefix)
    captions_dir = os.path.join(base_dir, captions_prefix)
    models_dir = os.path.join(base_dir, "models")

    os.makedirs(models_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.png"))
    )

    prompt_prefix = f"{prompt_prefix} " if len(prompt_prefix) > 0 else prompt_prefix
    prompt_suffix = f" {prompt_suffix}" if len(prompt_suffix) > 0 else prompt_suffix

    model_qnt_config = {}

    if use_auto_device_map:
        device_map = "auto"

    else:
        config = AutoConfig.from_pretrained(hf_model_id)
        max_memory = get_max_memory()

        with init_empty_weights():
            model = InstructBlipForConditionalGeneration(config)
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=[
                    "InstructBlipVisionModel",
                    "LlamaDecoderLayer",
                ],
                max_memory=max_memory,
            )

        device_map["language_model.lm_head"] = device_map[
            "language_model.model.embed_tokens"
        ]

        device_types = set(device_map.values())
        if "cpu" in device_types or "disk" in device_types:
            model_qnt_config.update(
                llm_int8_enable_fp32_cpu_offload=True, offload_folder=models_dir
            )

    if load_in_4bit:
        model_qnt_config["load_in_4bit"] = True
    elif load_in_8bit:
        model_qnt_config["load_in_8bit"] = True
    else:
        model_qnt_config["torch_dtype"] = torch.float16

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = InstructBlipForConditionalGeneration.from_pretrained(
        hf_model_id,
        cache_dir=models_dir,
        device_map=device_map,
        **model_qnt_config,
    )
    processor = AutoProcessor.from_pretrained(hf_model_id, cache_dir=models_dir)

    logger.debug(
        "Model memory footprint: %.1fGB", model.get_memory_footprint() / 1024**3
    )

    images_prefix = os.path.basename(images_dir)

    with open(
        os.path.join(captions_dir, "metadata.jsonl"), "w", encoding="utf-8"
    ) as output_file:
        for image_path in tqdm(image_paths):
            image = Image.open(image_path).convert("RGB")
            filename = os.path.basename(image_path)

            inputs = processor(
                images=image, text=caption_prompt, return_tensors="pt"
            ).to(device, torch.float16)

            outputs = model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
            )
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[
                0
            ].strip()

            metadata = {
                "file_name": f"{images_prefix}/{filename}",
                "text": f"{prompt_prefix}{generated_text}{prompt_suffix}".strip(),
            }

            output_file.write(json.dumps(metadata) + "\n")
            logger.debug("The line has been written: %s", metadata)


if __name__ == "__main__":
    logger.info("The image captioning task started...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/opt/ml/processing")
    parser.add_argument("--images-prefix", type=str, default="images")
    parser.add_argument("--captions-prefix", type=str, default="image_captions")
    parser.add_argument("--caption-prompt", type=str, default="Describe the photo.")
    parser.add_argument("--prompt-prefix", type=str, default="")
    parser.add_argument("--prompt-suffix", type=str, default="")
    parser.add_argument("--use-auto-device-map", type=arg_as_bool, default=False)
    parser.add_argument("--load-in-4bit", type=arg_as_bool, default=False)
    parser.add_argument("--load-in-8bit", type=arg_as_bool, default=False)

    args, _ = parser.parse_known_args()

    caption_images(
        args.base_dir,
        args.images_prefix,
        args.captions_prefix,
        HfModelId.INSTRUCT_BLIP.value,
        args.caption_prompt,
        args.prompt_prefix,
        args.prompt_suffix,
        args.use_auto_device_map,
        args.load_in_4bit,
        args.load_in_8bit,
        **MODEL_GEN_CONFIG,
    )

    logger.info("The image captioning task ended successfully.")
