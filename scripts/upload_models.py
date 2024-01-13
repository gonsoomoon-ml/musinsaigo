import os
import re
import sys
from shutil import copyfile
from typing import Final
import boto3
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from huggingface_hub import create_repo, upload_folder

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir))
)
from utils.config_handler import load_config
from utils.enums import DirName, FileName, HfModelId
from utils.logger import logger
from utils.misc import create_bucket_if_not_exists, decompress_file
from utils.torch_utils import bin_to_safetensors, convert_lora_safetensor_to_diffusers

CROSS_ATTENTION_SCALE: Final = 0.75


def find_latest_checkpoint(folder_path: str) -> str:
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


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            DirName.CONFIGS.value,
            FileName.CONFIG.value,
        )
    )
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

    src_model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, config.models_prefix, "src")
    )

    os.makedirs(src_model_dir, exist_ok=True)

    zip_file_path = os.path.join(src_model_dir, "model.tar.gz")

    s3_client.download_file(
        bucket,
        f"{config.base_prefix}/{config.models_prefix}/{config.model_data}/output/model.tar.gz",
        zip_file_path,
    )
    decompress_file(zip_file_path, src_model_dir, compression="tar")
    os.remove(zip_file_path)

    if config.use_sdxl:
        tgt_model_dir = src_model_dir

    else:
        tgt_model_dir = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, config.models_prefix, "tgt"
            )
        )

        bin_path = os.path.join(src_model_dir, "pytorch_lora_weights.bin")
        if not os.path.exists(bin_path):
            bin_path = os.path.join(
                src_model_dir,
                find_latest_checkpoint(src_model_dir),
                "pytorch_model.bin",
            )

        safetensors_path = os.path.join(
            src_model_dir, "pytorch_lora_weights.safetensors"
        )

        bin_to_safetensors(bin_path, safetensors_path)

        model = StableDiffusionPipeline.from_pretrained(
            HfModelId.SD_V1_5.value, torch_dtype=torch.float32
        )
        model.vae = AutoencoderKL.from_pretrained(
            HfModelId.SD_VAE.value, torch_dtype=torch.float32
        )
        model.scheduler = model.scheduler = EulerDiscreteScheduler.from_config(
            model.scheduler.config, use_karras_sigmas=True
        )

        model = convert_lora_safetensor_to_diffusers(
            model, src_model_dir, cross_attention_scale=CROSS_ATTENTION_SCALE
        )

        model.save_pretrained(tgt_model_dir)

    doc_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.curdir, "README.md")
    )
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
        create_pr=False,
        ignore_patterns=["step_*", "epoch_*"],
    )

    logger.info(
        "Downloading the model from S3, merging the weights, and uploading it to the HuggingFace hub '%s' is complete.",
        config.hf_model_id,
    )
