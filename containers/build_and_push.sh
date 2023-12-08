#!/bin/bash

REPOSITORY_NAME=${REPOSITORY_NAME}
AWS_PUBLIC_ACCOUNT=${AWS_PUBLIC_ACCOUNT}
DOCKERFILE_PATH=${DOCKERFILE_PATH}
BUILD_ARG=${BUILD_ARG-""}

CURRENT_DIRECTORY=$(pwd -P)

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
AWS_REGION=${AWS_REGION:-us-east-1}

ECR_REPO_FULLNAME="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/$REPOSITORY_NAME:latest"

if ! aws ecr describe-repositories --repository-names "$REPOSITORY_NAME" > /dev/null 2>&1; then
    aws ecr create-repository --repository-name "$REPOSITORY_NAME" > /dev/null
fi

aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REPO_FULLNAME}"
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "$AWS_PUBLIC_ACCOUNT.dkr.ecr.${AWS_REGION}.amazonaws.com"

if [ -n "$BUILD_ARG" ]; then
    eval "docker build $BUILD_ARG -f ${CURRENT_DIRECTORY}/$DOCKERFILE_PATH -t ${REPOSITORY_NAME} ."
else
    docker build -f "${CURRENT_DIRECTORY}/$DOCKERFILE_PATH" -t "${REPOSITORY_NAME}" .
fi

docker tag "${REPOSITORY_NAME}" "${ECR_REPO_FULLNAME}"
docker push "${ECR_REPO_FULLNAME}"