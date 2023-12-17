from enum import Enum


class BaseJobName(str, Enum):
    IMAGE_CAPTION: str = "image-caption"
    IMAGE_OUTPAINT: str = "image-outpaint"
    FINETUNE: str = "finetune"


class DirName(str, Enum):
    CONFIGS: str = "configs"
    LOGS: str = "logs"


class FileName(str, Enum):
    CONFIG: str = "config.yaml"
    LOG: str = "log.txt"


class HfModelId(str, Enum):
    SD_V1_5: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    SD_VAE: str = "stabilityai/sd-vae-ft-mse"
    SD_INPAINT: str = "runwayml/stable-diffusion-inpainting"
    SDXL_V1_0_BASE: str = "stabilityai/stable-diffusion-xl-base-1.0"
    SDXL_V1_0_REFINER: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
    SDXL_VAE: str = "madebyollin/sdxl-vae-fp16-fix"
    INSTRUCT_BLIP: str = "Salesforce/instructblip-vicuna-7b"


class Version(str, Enum):
    PYTHON: str = "py310"
    PYTORCH: str = "2.0.0"
    TRANSFORMERS: str = "4.28.1"
