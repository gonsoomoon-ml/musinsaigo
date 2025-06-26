FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04

ENV PYTHONUNBUFFERED=TRUE

# diffusers 없이 필요한 패키지만 추가
RUN pip3 install --no-cache-dir \
    peft==0.7.1 \
    wandb>=0.16.1 \
    xformers>=0.0.19

ARG CODE_DIR
ARG CODE_FILENAME
ARG SCRIPT_ARGS
ARG LAUNCH_ARGS=""
ARG WANDB_API_KEY=""

ENV CODE_FILENAME ${CODE_FILENAME}
ENV SCRIPT_ARGS ${SCRIPT_ARGS}
ENV LAUNCH_ARGS ${LAUNCH_ARGS}
ENV WANDB_API_KEY ${WANDB_API_KEY}

COPY configs/accelerate_config.yaml /opt/ml/code/accelerate_config.yaml
COPY configs/config.yaml /opt/ml/code/configs/config.yaml
COPY containers/entrypoint.sh /opt/ml/code/entrypoint.sh
COPY ${CODE_DIR}/${CODE_FILENAME} /opt/ml/code/${CODE_FILENAME}

RUN chmod +x /opt/ml/code/entrypoint.sh

ENTRYPOINT ["/opt/ml/code/entrypoint.sh"]