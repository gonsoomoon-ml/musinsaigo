from dataclasses import dataclass
from typing import Any, Optional
import yaml
from utils.logger import logger


def get_default(value: Any, default: Any) -> Any:
    return default if value is None else value


def validate(value: Any, key: str = Optional[str]) -> Any:
    key = "" if key is None else f", {key}"
    if value is None:
        raise ValueError(f"The argument{key} must have a value.")
    return value


@dataclass
class Config:
    profile_name: Optional[str]
    region_name: str
    role: Optional[str]
    bucket: Optional[str]
    base_prefix: str
    dataset_prefix: str
    images_prefix: str
    models_prefix: str
    unsplash_api_key: Optional[str]
    hf_token: Optional[str]
    wandb_api_key: Optional[str]
    data_source: str
    num_images: int
    data_query: Optional[str]
    is_street_snap: bool
    order_method: Optional[str]
    is_best: bool
    caption_prompt: str
    prompt_prefix: str
    prompt_suffix: str
    outpaint_images: bool
    outpaint_prompt: str
    data_prep_repo: str
    model_train_repo: str
    model_data: Optional[str]
    caption_instance_type: str
    outpaint_instance_type: str
    train_instance_type: str
    infer_instance_type: str
    use_multi_gpus: bool
    use_sdxl: bool
    use_dreambooth: bool
    num_train_epochs: Optional[int]
    resolution: int
    subject_name: str
    class_name: str
    center_crop: bool
    random_flip: bool
    train_text_encoder: bool
    batch_size: int
    max_train_steps: Optional[int]
    learning_rate: float
    lr_scheduler: str
    checkpointing_steps: int
    push_to_hub: bool
    hf_model_id: str
    reduce_memory_usage: bool
    validation_prompt: str
    endpoint_name: str


def load_config(config_path: str) -> Config:
    with open(config_path, encoding="utf-8") as file_path:
        config = yaml.safe_load(file_path)

    base_prefix = validate(config["environment"]["s3_base_prefix"], "s3_base_prefix")
    model_data = config["model"]["sm_model_data"]
    num_train_epochs = config["model"]["num_train_epochs"]
    max_train_steps = config["model"]["max_train_steps"]
    use_sdxl = get_default(config["model"]["use_sdxl"], False)
    use_dreambooth = get_default(config["model"]["use_dreambooth"], False)
    subject_name = get_default(config["model"]["subject_name"], "sks")
    class_name = get_default(config["model"]["class_name"], "fashion street snap")
    checkpointing_steps = get_default(config["model"]["checkpointing_steps"], 500)
    validation_prompt = get_default(
        config["model"]["validation_prompt"],
        "a photo of a woman wearing a white t-shirt and jeans",
    )

    if num_train_epochs is None and max_train_steps is None:
        num_train_epochs = 100

    if model_data is None and use_sdxl:
        if max_train_steps:
            logger.warning("The GPU memory issue changes checkpointing steps.")
            checkpointing_steps = max_train_steps

    if (
        model_data is None
        and use_dreambooth
        and subject_name not in validation_prompt
        and class_name not in validation_prompt
    ):
        logger.warning(
            "The validation prompt changes because it doesn't include the subject and class name."
        )
        validation_prompt += f", {subject_name} {class_name}"

    return Config(
        profile_name=config["environment"]["iam_profile_name"],
        region_name=get_default(config["environment"]["region_name"], "us-east-1"),
        role=config["environment"]["iam_role"],
        bucket=config["environment"]["s3_bucket"],
        base_prefix=base_prefix,
        dataset_prefix=get_default(
            config["environment"]["s3_dataset_prefix"], "dataset"
        ),
        images_prefix=get_default(config["environment"]["s3_images_prefix"], "images"),
        models_prefix=get_default(config["environment"]["s3_models_prefix"], "models"),
        unsplash_api_key=config["environment"]["unsplash_api_key"],
        hf_token=config["environment"]["hf_token"],
        wandb_api_key=config["environment"]["wandb_api_key"],
        data_source=get_default(config["data"]["data_source"], "unsplash"),
        num_images=get_default(config["data"]["num_images"], 100),
        data_query=config["data"]["data_query"],
        is_street_snap=get_default(config["data"]["is_street_snap"], True),
        order_method=get_default(config["data"]["order_method"], None),
        is_best=get_default(config["data"]["is_best"], True),
        caption_prompt=get_default(
            config["data"]["caption_prompt"], "Describe the photo."
        ),
        prompt_prefix=get_default(config["data"]["prompt_prefix"], ""),
        prompt_suffix=get_default(config["data"]["prompt_suffix"], ""),
        outpaint_images=get_default(config["data"]["outpaint_images"], False),
        outpaint_prompt=get_default(config["data"]["outpaint_prompt"], ""),
        data_prep_repo=validate(
            config["model"]["ecr_data_prep_repo"], "ecr_data_prep_repo"
        ),
        model_train_repo=validate(
            config["model"]["ecr_model_train_repo"], "ecr_model_train_repo"
        ),
        model_data=model_data,
        caption_instance_type=get_default(
            config["model"]["sm_caption_instance_type"], "ml.g4dn.xlarge"
        ),
        outpaint_instance_type=get_default(
            config["model"]["sm_outpaint_instance_type"], "ml.g4dn.xlarge"
        ),
        train_instance_type=get_default(
            config["model"]["sm_train_instance_type"], "ml.g4dn.xlarge"
        ),
        infer_instance_type=get_default(
            config["model"]["sm_infer_instance_type"], "ml.g4dn.xlarge"
        ),
        use_multi_gpus=get_default(config["model"]["use_multi_gpus"], False),
        use_sdxl=use_sdxl,
        use_dreambooth=use_dreambooth,
        num_train_epochs=num_train_epochs,
        resolution=get_default(config["model"]["resolution"], 512),
        subject_name=subject_name,
        class_name=class_name,
        center_crop=get_default(config["model"]["center_crop"], False),
        random_flip=get_default(config["model"]["random_flip"], False),
        train_text_encoder=get_default(config["model"]["train_text_encoder"], False),
        batch_size=get_default(config["model"]["batch_size"], 16),
        max_train_steps=max_train_steps,
        learning_rate=get_default(config["model"]["learning_rate"], 1e-04),
        lr_scheduler=get_default(config["model"]["lr_scheduler"], "cosine"),
        checkpointing_steps=checkpointing_steps,
        push_to_hub=get_default(config["model"]["push_to_hub"], False),
        hf_model_id=get_default(config["model"]["hf_model_id"], base_prefix),
        reduce_memory_usage=get_default(config["model"]["reduce_memory_usage"], False),
        validation_prompt=validation_prompt,
        endpoint_name=get_default(config["model"]["sm_endpoint_name"], base_prefix),
    )
