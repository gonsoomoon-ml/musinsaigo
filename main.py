import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import ScriptProcessor
from utils.config_handler import Config, load_config
from utils.logger import logger
from utils.misc import (
    change_dict_to_cli_args,
    create_bucket_if_not_exists,
    create_role_if_not_exists,
    get_job_name,
    make_ecr_uri,
    make_s3_uri,
    run_shell_script,
)


CONFIG_PREFIX = "configs"
CONFIG_FILENAME = "config.yaml"
CAPTIONS_PREFIX = "image_captions"

HF_MODEL_IDS = [
    "SG161222/Realistic_Vision_V5.1_noVAE",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "madebyollin/sdxl-vae-fp16-fix",
]
TRANSFORMERS_VERSION = "4.28.1"
PYTORCH_VERSION = "2.0.0"
PY_VERSION = "py310"

BASE_JOB_NAMES = ["image-caption", "image-outpaint", "finetune"]
PROC_BASE_DIR = "/opt/ml/processing"

SKIP_DATA_PREP = True


def main(config: Config) -> None:
    boto_session = boto3.Session(
        profile_name=config.profile_name, region_name=config.region_name
    )
    account_id = boto_session.client("sts").get_caller_identity().get("Account")
    sm_session = sagemaker.session.Session(boto_session=boto_session)

    role = (
        create_role_if_not_exists(boto_session, config.region_name, logger=logger)
        if config.role is None
        else config.role
    )
    bucket = (
        create_bucket_if_not_exists(boto_session, config.region_name, logger=logger)
        if config.bucket is None
        else config.bucket
    )

    raw_dataset_uri = make_s3_uri(
        bucket, f"{config.base_prefix}/raw_{config.dataset_prefix}"
    )
    proc_dataset_uri = make_s3_uri(
        bucket, f"{config.base_prefix}/proc_{config.dataset_prefix}"
    )
    dataset_uri = proc_dataset_uri if config.outpaint_images else raw_dataset_uri
    models_uri = make_s3_uri(bucket, f"{config.base_prefix}/{config.models_prefix}")

    if config.use_dreambooth:
        train_prefix = "sdxl_dreambooth" if config.use_sdxl else "sd_dreambooth"
    else:
        train_prefix = "sdxl_lora" if config.use_sdxl else "sd_lora"

    infer_prefix = "sdxl" if config.use_sdxl else "sd"

    if config.model_data is None:
        if not SKIP_DATA_PREP:
            run_shell_script(
                os.path.join("containers", "build_and_push.sh"),
                {
                    "REPOSITORY_NAME": config.data_prep_repo,
                    "AWS_PUBLIC_ACCOUNT": "763104351884",
                    "DOCKERFILE_PATH": os.path.join(
                        "containers", "data_prep", "Dockerfile"
                    ),
                },
            )

            data_prep_image_uri = make_ecr_uri(
                account_id, config.region_name, config.data_prep_repo
            )

            arguments = [
                "--base-dir",
                PROC_BASE_DIR,
                "--images-prefix",
                config.images_prefix,
                "--captions-prefix",
                CAPTIONS_PREFIX,
            ]
            if len(config.prompt_prefix) > 0:
                arguments.extend(["--prompt-prefix", config.prompt_prefix])

            if len(config.prompt_suffix) > 0:
                arguments.extend(["--prompt-suffix", config.prompt_suffix])

            script_processor = ScriptProcessor(
                role=role,
                image_uri=data_prep_image_uri,
                instance_type=config.caption_instance_type,
                instance_count=1,
                command=["python3"],
                max_runtime_in_seconds=259200,
                base_job_name=BASE_JOB_NAMES[0],
                sagemaker_session=sm_session,
            )

            script_processor.run(
                inputs=[
                    ProcessingInput(
                        source=f"{raw_dataset_uri}/{config.images_prefix}",
                        destination=f"{PROC_BASE_DIR}/{config.images_prefix}",
                        input_name="images",
                    ),
                ],
                outputs=[
                    ProcessingOutput(
                        source=f"{PROC_BASE_DIR}/{CAPTIONS_PREFIX}",
                        destination=raw_dataset_uri,
                        output_name="image_captions",
                    )
                ],
                code=os.path.join(
                    "code", "data_prep", "image_caption", "image_caption.py"
                ),
                arguments=arguments,
                logs=True,
                job_name=get_job_name(BASE_JOB_NAMES[0]),
            )

            if config.outpaint_images:
                arguments = [
                    "--base-dir",
                    PROC_BASE_DIR,
                    "--dataset-prefix",
                    config.dataset_prefix,
                    "--images-prefix",
                    config.images_prefix,
                    "--resolution",
                    str(config.resolution),
                ]
                if len(config.outpaint_prompt) > 0:
                    arguments.extend(["--outpaint-prompt", config.outpaint_prompt])

                script_processor = ScriptProcessor(
                    role=role,
                    image_uri=data_prep_image_uri,
                    instance_type=config.outpaint_instance_type,
                    instance_count=1,
                    command=["python3"],
                    max_runtime_in_seconds=259200,
                    base_job_name=BASE_JOB_NAMES[1],
                    sagemaker_session=sm_session,
                )

                script_processor.run(
                    inputs=[
                        ProcessingInput(
                            source=raw_dataset_uri,
                            destination=f"{PROC_BASE_DIR}/raw_{config.dataset_prefix}",
                            input_name="raw_dataset",
                        ),
                    ],
                    outputs=[
                        ProcessingOutput(
                            source=f"{PROC_BASE_DIR}/proc_{config.dataset_prefix}",
                            destination=dataset_uri,
                            output_name="proc_dataset",
                        )
                    ],
                    code=os.path.join(
                        "code", "data_prep", "image_outpaint", "image_outpaint.py"
                    ),
                    arguments=arguments,
                    logs=True,
                    job_name=get_job_name(BASE_JOB_NAMES[1]),
                )

        build_arg = f"--build-arg CODE_DIR={os.path.join('code', 'model_train', train_prefix)} \
        --build-arg CODE_FILENAME=train.py"

        if config.use_multi_gpus:
            build_arg += " --build-arg LAUNCH_ARGS='--config_file \
            /opt/ml/code/accelerate_config.yaml'"

        hyperparameters = {
            "pretrained_model_name_or_path": HF_MODEL_IDS[1]
            if config.use_sdxl
            else HF_MODEL_IDS[0],
            "dataloader_num_workers": 8,
            "resolution": min(config.resolution, 1024)
            if config.use_sdxl
            else min(config.resolution, 768),
            "train_batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "max_grad_norm": 1,
            "lr_scheduler": config.lr_scheduler,
            "lr_warmup_steps": 0,
            "checkpointing_steps": config.checkpointing_steps,
            "seed": 42,
        }

        if config.num_train_epochs is None:
            hyperparameters["max_train_steps"] = config.max_train_steps
        else:
            hyperparameters["num_train_epochs"] = config.num_train_epochs

        if config.center_crop:
            hyperparameters["center_crop"] = ""

        if not config.use_dreambooth and config.random_flip:
            hyperparameters["random_flip"] = ""

        if config.use_dreambooth:
            hyperparameters[
                "instance_prompt"
            ] = f"'a photo of {config.subject_name} {config.class_name}'"
            hyperparameters["num_class_images"] = 200
            hyperparameters["with_prior_preservation"] = ""
            hyperparameters["class_prompt"] = f"'a photo of {config.class_name}'"

        if config.use_dreambooth or config.use_sdxl:
            hyperparameters["train_text_encoder"] = ""

        if config.reduce_memory_usage:
            hyperparameters["gradient_accumulation_steps"] = 4
            hyperparameters["gradient_checkpointing"] = ""
            hyperparameters["enable_xformers_memory_efficient_attention"] = ""

            if not config.use_multi_gpus:
                hyperparameters["use_8bit_adam"] = ""

            if config.use_sdxl:
                hyperparameters["pretrained_vae_model_name_or_path"] = HF_MODEL_IDS[2]
                hyperparameters["mixed_precision"] = "fp16"

        if config.wandb_api_key:
            prompt_prefix = (
                f"{config.prompt_prefix} "
                if len(config.prompt_prefix) > 0
                else config.prompt_prefix
            )
            prompt_suffix = (
                f" {config.prompt_suffix}"
                if len(config.prompt_suffix) > 0
                else config.prompt_suffix
            )
            validation_prompt = (
                f"'{prompt_prefix}{config.validation_prompt}{prompt_suffix}'"
            )

            hyperparameters["report_to"] = "wandb"
            hyperparameters["validation_prompt"] = validation_prompt.strip()

            build_arg += f" --build-arg WANDB_API_KEY={config.wandb_api_key}"

        if config.push_to_hub and config.hf_model_id:
            hyperparameters["push_to_hub"] = ""
            hyperparameters["hub_token"] = config.hf_token
            hyperparameters["hub_model_id"] = config.hf_model_id

        build_arg += (
            f' --build-arg SCRIPT_ARGS="{change_dict_to_cli_args(hyperparameters)}"'
        )

        run_shell_script(
            os.path.join("containers", "build_and_push.sh"),
            {
                "REPOSITORY_NAME": config.model_train_repo,
                "AWS_PUBLIC_ACCOUNT": "763104351884",
                "DOCKERFILE_PATH": os.path.join(
                    "containers", "model_train", "Dockerfile"
                ),
                "BUILD_ARG": build_arg,
            },
        )

        model_train_image_uri = make_ecr_uri(
            account_id, config.region_name, config.model_train_repo
        )

        estimator = Estimator(
            image_uri=model_train_image_uri,
            role=role,
            instance_count=1,
            instance_type=config.train_instance_type,
            max_run=259200,
            output_path=models_uri,
            base_job_name=get_job_name(BASE_JOB_NAMES[2], prefix=train_prefix),
            sagemaker_session=sm_session,
        )

        _ = estimator.fit(
            {
                "training": f"{dataset_uri}/{config.images_prefix}"
                if config.use_dreambooth
                else f"{dataset_uri}/"
            },
            logs=True,
        )

        model_data = estimator.model_data

    else:
        model_data = make_s3_uri(
            bucket,
            f"{config.base_prefix}/{config.models_prefix}/{config.model_data}/output",
            filename="model.tar.gz",
        )

    model = HuggingFaceModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PY_VERSION,
        source_dir=os.path.join("code", "inference", infer_prefix),
        sagemaker_session=sm_session,
    )

    _ = model.deploy(
        initial_instance_count=1,
        instance_type=config.infer_instance_type,
        endpoint_name=config.endpoint_name,
    )

    logger.info(
        "Fine-tuning the image generative model and deploying an endpoint to %s is complete.",
        config.endpoint_name,
    )


if __name__ == "__main__":
    logger.info("The image generative model fine-tuning and deployment job started...")

    config_path = os.path.join(CONFIG_PREFIX, CONFIG_FILENAME)
    config = load_config(config_path)

    main(config)

    logger.info(
        "The image generative model fine-tuning and deployment job ended successfully."
    )
