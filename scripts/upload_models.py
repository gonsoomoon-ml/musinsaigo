import os
import re
from shutil import copyfile
from typing import Optional
import boto3
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from huggingface_hub import create_repo, upload_folder
from utils.config_handler import load_config
from utils.logger import logger
from utils.misc import create_bucket_if_not_exists, decompress_file
from utils.torch_utils import bin_to_safetensors, convert_lora_safetensor_to_diffusers


def find_latest_checkpoint(folder_path: str) -> Optional[str]:
    entries = os.listdir(folder_path)
    checkpoint_folders = [
        entry
        for entry in entries
        if entry.startswith("checkpoint-")
        and os.path.isdir(os.path.join(folder_path, entry))
    ]

    checkpoint_numbers = [
        int(re.search(r"checkpoint-(\d+)", folder).group(1))
        for folder in checkpoint_folders
    ]

    if not checkpoint_numbers:
        return None

    latest_checkpoint_number = max(checkpoint_numbers)

    return f"checkpoint-{latest_checkpoint_number}"


CONFIG_PREFIX = "configs"
CONFIG_FILENAME = "config.yaml"

HF_MODEL_IDS = [
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "stabilityai/sd-vae-ft-mse",
]
CROSS_ATTENTION_SCALE = 0.75


if __name__ == "__main__":
    config_path = os.path.join("..", CONFIG_PREFIX, CONFIG_FILENAME)
    config = load_config(config_path)

    boto_session = boto3.Session(
        profile_name=config.profile_name, region_name=config.region_name
    )
    s3_client = boto_session.client("s3")
    bucket = (
        create_bucket_if_not_exists(boto_session, config.region_name, logger=logger)
        if config.bucket is None
        else config.bucket
    )

    src_model_dir = os.path.join("..", config.models_prefix, "src")
    tgt_model_dir = os.path.join("..", config.models_prefix, "tgt")

    os.makedirs(src_model_dir, exist_ok=True)

    zip_file_path = os.path.join(src_model_dir, "model.tar.gz")

    s3_client.download_file(
        bucket,
        f"{config.base_prefix}/{config.models_prefix}/{config.model_data}/output/model.tar.gz",
        zip_file_path,
    )
    decompress_file(zip_file_path, src_model_dir, compression="tar")

    bin_path = os.path.join(src_model_dir, "pytorch_lora_weights.bin")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(
            src_model_dir, find_latest_checkpoint(src_model_dir), "pytorch_model.bin"
        )

    safetensors_path = os.path.join(src_model_dir, "pytorch_lora_weights.safetensors")

    bin_to_safetensors(bin_path, safetensors_path)

    model = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_IDS[0], torch_dtype=torch.float32
    )
    model.vae = AutoencoderKL.from_pretrained(
        HF_MODEL_IDS[1], torch_dtype=torch.float32
    )
    model.scheduler = model.scheduler = EulerDiscreteScheduler.from_config(
        model.scheduler.config, use_karras_sigmas=True
    )

    model = convert_lora_safetensor_to_diffusers(
        model, src_model_dir, cross_attention_scale=CROSS_ATTENTION_SCALE
    )

    model.save_pretrained(tgt_model_dir)

    doc_path = os.path.join(".", "README.md")
    if os.path.exists(doc_path):
        copyfile(doc_path, os.path.join(tgt_model_dir, "README.md"))

    _ = create_repo(
        repo_id=config.hf_model_id,
        token=config.hf_token,
        exist_ok=True,
    )

    _ = upload_folder(
        repo_id=config.hf_model_id,
        folder_path=tgt_model_dir,
        commit_message="End of training",
        token=config.hf_token,
        create_pr=True,
        ignore_patterns=["step_*", "epoch_*"],
    )

    logger.info(
        "Downloading the model from S3, merging the weights, and uploading it to the HuggingFace hub '%s' is complete.",
        config.hf_model_id,
    )
