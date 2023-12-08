import argparse
import base64
import json
import logging
import os
import subprocess
import tarfile
import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional, Union
import boto3
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sagemaker.utils import unique_name_from_base


def arg_as_bool(value: Any):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError


def compress_dir_to_model_tar_gz(
    tar_dir: Optional[str] = None,
    output_file: str = "model.tar.gz",
    logger: Optional[logging.Logger] = None,
) -> None:
    parent_dir = os.getcwd()
    os.chdir(tar_dir)

    log_or_print("The following directories and files will be compressed.", logger)

    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir("."):
            log_or_print(item, logger)
            tar.add(item, arcname=item)

    os.chdir(parent_dir)


def change_dict_to_cli_args(input_dict: Dict[str, Union[float, int, str]]) -> str:
    args_list = []
    for key, value in input_dict.items():
        if value == "":
            args_list.append(f"--{key}")
        else:
            args_list.append(f"--{key} {value}")
    return " ".join(args_list)


def create_bucket_if_not_exists(
    boto_session: boto3.session.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    s3_client = boto_session.client("s3")
    sts_client = boto_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    bucket_name = f"sagemaker-{region_name}-{account_id}"

    try:
        s3_client.head_bucket(Bucket=bucket_name)
        msg = f"The following S3 bucket was found: {bucket_name}"

    except s3_client.exceptions.NoSuchBucket:
        s3_client.create_bucket(Bucket=bucket_name)
        msg = f"The following S3 bucket was created: {bucket_name}"

    log_or_print(msg, logger)
    return bucket_name


def create_role_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    iam_client = boto_session.client("iam")

    role_name = f"AmazonSageMaker-ExecutionRole-{region_name}"
    try:
        role = iam_client.get_role(RoleName=role_name)
        msg = f"The following IAM role was found: {role['Role']['Arn']}"

    except iam_client.exceptions.NoSuchEntityException:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="SageMaker Execution Role",
        )
        policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        msg = f"The following IAM role was created: {role['Role']['Arn']}"

    log_or_print(msg, logger)
    return role_name


def download_dir_from_s3(
    boto_session: boto3.Session, local_dir: str, bucket: str, prefix: str
) -> None:
    s3_bucket = boto_session.resource("s3").Bucket(bucket)
    for obj in s3_bucket.objects.filter(Prefix=prefix):
        os.makedirs(local_dir, exist_ok=True)
        s3_bucket.download_file(
            obj.key, os.path.join(local_dir, obj.key.split("/")[-1])
        )


def encode_base64_image(file_name: str) -> str:
    with open(file_name, "rb") as image:
        image_string = base64.b64encode(image.read()).decode()
    return image_string


def decode_base64_image(image_string: str) -> Any:
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


def decompress_file(
    source_file_path: str, target_dir: str, compression: str = "zip"
) -> None:
    if compression == "zip":
        with zipfile.ZipFile(source_file_path) as file:
            file.extractall(target_dir)
    elif compression == "tar":
        with tarfile.open(source_file_path) as file:
            file.extractall(target_dir)
    else:
        raise ValueError("The argument, 'compression' should be 'zip or 'tar'.")


def delete_files_in_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")

    objects_to_delete = [
        {"Key": obj.key}
        for obj in boto_session.resource("s3")
        .Bucket(bucket_name)
        .objects.filter(Prefix=prefix)
    ]
    s3_client.delete_objects(Bucket=bucket_name, Delete={"Objects": objects_to_delete})

    for obj in objects_to_delete:
        log_or_print(
            f"The 's3://{bucket_name}/{obj['Key']}' file has been deleted.", logger
        )


def display_images(
    images: List[Image.Image],
    columns: int = 3,
    fig_size: tuple = (15, 15),
) -> None:
    total_images = len(images)
    rows = total_images // columns + 1
    plt.figure(figsize=fig_size)
    for i, image in enumerate(images, start=1):
        plt.subplot(rows, columns, i)
        plt.imshow(image)
        plt.axis("off")
    plt.show()


def display_image_grid(images: List[Image.Image], rows: int, cols: int) -> Image.Image:
    assert len(images) == rows * cols
    width, height = images[0].size
    grid = Image.new("RGB", size=(cols * width, rows * height))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * width, i // cols * height))
    return grid


def get_job_name(base_job_name: str, prefix: str = "", suffix: str = "") -> str:
    prefix = f"{prefix}-" if len(prefix) > 0 else prefix
    suffix = f"-{suffix}" if len(suffix) > 0 else suffix
    job_name = unique_name_from_base(
        base_job_name, max_length=63 - len(prefix) - len(suffix)
    )
    return f"{prefix}{job_name}{suffix}".replace("_", "-").replace("/", "-")


def get_max_memory() -> Dict[int, str]:
    return {
        i: f"{round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3) - 1}GiB"
        for i in range(torch.cuda.device_count())
    }


def log_or_print(msg: str, logger: Optional[logging.Logger] = None) -> None:
    if logger:
        logger.info(msg)
    else:
        print(msg)


def make_ecr_uri(account_id: str, region_name: str, repo_name: str) -> str:
    return f"{account_id}.dkr.ecr.{region_name}.amazonaws.com/{repo_name}"


def make_s3_uri(bucket: str, prefix: str, filename: Optional[str] = None) -> str:
    prefix = prefix if filename is None else os.path.join(prefix, filename)
    return f"s3://{bucket}/{prefix}"


def run_shell_script(file_path: str, env_vars: Optional[Dict[str, str]] = None) -> None:
    if env_vars:
        env = {**env_vars, **dict(os.environ)}
    else:
        env = {**dict(os.environ)}
    subprocess.run(["bash", file_path], env=env, check=True)


def upload_dir_to_s3(
    boto_session: boto3.Session,
    local_dir: str,
    bucket: str,
    prefix: str,
    file_ext_to_excl: Optional[List[str]] = None,
    public_readable: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")
    file_ext_to_excl = file_ext_to_excl or []

    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.split(".")[-1] not in file_ext_to_excl:
                file_path = os.path.join(root, file)
                s3_path = os.path.join(prefix, os.path.relpath(file_path, local_dir))

                extra_args = {"ACL": "public-read"} if public_readable else {}
                s3_client.upload_file(file_path, bucket, s3_path, ExtraArgs=extra_args)

                log_or_print(
                    f"The '{file_path}' file has been uploaded to 's3://{bucket}/{s3_path}'.",
                    logger,
                )
