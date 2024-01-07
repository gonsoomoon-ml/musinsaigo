FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.0.0-transformers4.28.1-gpu-py310-cu118-ubuntu20.04

ENV PYTHONUNBUFFERED=TRUE

RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    accelerate>=0.25.0 \
    bitsandbytes>=0.41.0 \
    diffusers==0.25.0 \
    safetensors>=0.4.1 \
    torch==2.0.0 \
    transformers==4.36.2 \
    xformers>=0.0.19

COPY ./utils/ /tmp/utils/

ENTRYPOINT ["python3"]