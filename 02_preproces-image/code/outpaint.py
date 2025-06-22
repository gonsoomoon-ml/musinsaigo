#!/usr/bin/env python3
"""
Standalone Image Outpainting Script for Musinsaigo
이 스크립트는 무신사 패션 이미지들을 AI로 확장하여 정사각형 이미지로 만듭니다.

사용법:
    python outpaint.py --input-dir /path/to/raw_dataset --output-dir /path/to/proc_dataset
"""

import argparse
import json
import os
import sys
from shutil import copyfile
from typing import Final, Optional
import numpy as np
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from tqdm import tqdm
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('outpaint.log')
    ]
)
logger = logging.getLogger(__name__)

# 모델 생성 설정
MODEL_GEN_CONFIG: Final = {
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "negative_prompt": "other people in the background",
    "seed": 42,
}

# 모델 ID
SD_INPAINT_MODEL = "runwayml/stable-diffusion-inpainting"


def arg_as_bool(value):
    """문자열을 boolean으로 변환"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}")


def get_loc(src_len: int, tgt_len: int) -> int:
    """이미지를 중앙에 배치하기 위한 위치 계산"""
    return round((src_len - tgt_len) // 2)


def outpaint_images(
    input_dir: str,
    output_dir: str,
    images_prefix: str = "images",
    outpaint_prompt: str = "",
    resolution: int = 1024,
    run_compile: bool = False,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    max_images: Optional[int] = None,  # 테스트용 최대 이미지 수
) -> None:
    """
    이미지들을 outpainting하여 정사각형으로 확장
    
    Args:
        input_dir: 원본 데이터셋 디렉토리 (raw_dataset)
        output_dir: 출력 데이터셋 디렉토리 (proc_dataset)
        images_prefix: 이미지 폴더명 (기본값: "images")
        outpaint_prompt: 확장할 내용 프롬프트
        resolution: 최종 해상도 (기본값: 1024)
        run_compile: PyTorch 컴파일 사용 여부
        num_inference_steps: 추론 단계 수
        guidance_scale: 프롬프트 준수 강도
        negative_prompt: 제외할 내용
        seed: 재현성을 위한 시드
        max_images: 테스트용 최대 이미지 수
    """
    
    # 디렉토리 설정
    raw_images_dir = os.path.join(input_dir, images_prefix)
    proc_images_dir = os.path.join(output_dir, images_prefix)
    
    # 출력 디렉토리 생성
    os.makedirs(proc_images_dir, exist_ok=True)
    
    # 메타데이터 파일 경로
    raw_metadata_path = os.path.join(input_dir, "metadata.jsonl")
    proc_metadata_path = os.path.join(output_dir, "metadata.jsonl")
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"사용 디바이스: {device}")
    
    if device == "cpu":
        logger.warning("GPU가 감지되지 않았습니다. CPU에서 실행됩니다 (느릴 수 있습니다).")
    
    # 모델 로드
    logger.info("Stable Diffusion Inpainting 모델을 로드합니다...")
    model = StableDiffusionInpaintPipeline.from_pretrained(
        SD_INPAINT_MODEL,
        torch_dtype=torch.float16,
    ).to(device)
    
    # PyTorch 컴파일 (선택사항)
    if run_compile:
        logger.info("PyTorch 컴파일을 적용합니다...")
        model.unet.to(memory_format=torch.channels_last)
        torch.compile(model.unet, mode="reduce-overhead", fullgraph=True)
    
    # 생성기 설정
    generator = torch.Generator(device=device).manual_seed(seed or MODEL_GEN_CONFIG["seed"])
    
    # 이미지 파일 목록 생성
    logger.info("이미지 파일 목록을 생성합니다...")
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    if os.path.exists(raw_images_dir):
        for filename in os.listdir(raw_images_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_files.append(filename)
    
    if not image_files:
        logger.error(f"이미지 파일을 찾을 수 없습니다: {raw_images_dir}")
        return
    
    # 테스트용으로 최대 이미지 수 제한
    if max_images:
        image_files = image_files[:max_images]
        logger.info(f"테스트 모드: {max_images}개 이미지만 처리합니다.")
    
    total_images = len(image_files)
    logger.info(f"총 {total_images}개의 이미지를 처리합니다.")
    
    # 이미지 outpainting 시작
    logger.info("이미지 outpainting을 시작합니다...")
    processed_count = 0
    error_count = 0
    processed_metadata = []
    
    for filename in tqdm(image_files, total=total_images, desc="Outpainting 진행률"):
        try:
            # 프롬프트 설정 (outpaint_prompt가 있으면 사용, 없으면 기본 프롬프트 사용)
            prompt = outpaint_prompt if len(outpaint_prompt) > 0 else "a photo of a person (people)"
            
            # 원본 이미지 로드
            src_image_path = os.path.join(raw_images_dir, filename)
            if not os.path.exists(src_image_path):
                logger.warning(f"이미지를 찾을 수 없습니다: {src_image_path}")
                error_count += 1
                continue
            
            src_image = Image.open(src_image_path).convert("RGB")
            src_image_size = np.array(src_image).shape
            
            # 정사각형 크기 계산
            max_len = max(src_image_size[0], src_image_size[1])
            tgt_image_size = max_len, max_len
            
            # RGBA 이미지 생성 (투명 배경)
            rgba_image = Image.new(mode="RGBA", size=tgt_image_size)
            
            # 원본 이미지를 중앙에 배치
            Image.Image.paste(
                rgba_image,
                src_image,
                (
                    get_loc(tgt_image_size[1], src_image_size[1]),
                    get_loc(tgt_image_size[0], src_image_size[0]),
                ),
            )
            
            # RGB로 변환
            rgb_image = rgba_image.convert("RGB")
            
            # 마스크 생성 (투명 영역을 마스크로 설정)
            full_mask = np.array(rgba_image)[:, :, 3] == 0
            full_mask = full_mask.astype(np.uint8) * 255
            full_mask = np.dstack([np.array(full_mask)] * 3)
            mask_image = Image.fromarray(full_mask)
            
            # Outpainting 실행
            tgt_image = model(
                prompt=prompt,
                height=resolution,
                width=resolution,
                image=rgb_image.resize((resolution, resolution)),
                mask_image=mask_image.resize((resolution, resolution)),
                generator=generator,
                num_inference_steps=num_inference_steps or MODEL_GEN_CONFIG["num_inference_steps"],
                guidance_scale=guidance_scale or MODEL_GEN_CONFIG["guidance_scale"],
                negative_prompt=negative_prompt or MODEL_GEN_CONFIG["negative_prompt"],
            ).images[0]
            
            # 원본 이미지 크기 계산
            tgt_image_size = round(
                resolution * src_image_size[0] / tgt_image_size[0]
            ), round(resolution * src_image_size[1] / tgt_image_size[1])
            
            # 생성된 이미지에 원본 이미지 복원
            Image.Image.paste(
                tgt_image,
                src_image.resize(
                    (
                        tgt_image_size[1],
                        tgt_image_size[0],
                    )
                ),
                (
                    get_loc(resolution, tgt_image_size[1]),
                    get_loc(resolution, tgt_image_size[0]),
                ),
            )
            
            # 결과 이미지 저장
            output_image_path = os.path.join(proc_images_dir, filename)
            tgt_image.save(output_image_path)
            
            # 메타데이터 생성
            metadata = {
                "file_name": f"{images_prefix}/{filename}",
                "text": prompt
            }
            processed_metadata.append(metadata)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생 ({filename}): {e}")
            error_count += 1
            continue
    
    # 메타데이터 파일 저장
    with open(proc_metadata_path, "w", encoding="utf-8") as output_file:
        for metadata in processed_metadata:
            output_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    
    # 결과 요약
    logger.info("=" * 50)
    logger.info(" Outpainting 작업이 완료되었습니다!")
    logger.info(f"✅ 성공적으로 처리된 이미지: {processed_count}개")
    if error_count > 0:
        logger.warning(f"⚠️ 처리 실패한 이미지: {error_count}개")
    logger.info(f"📁 입력 디렉토리: {input_dir}")
    logger.info(f"📁 출력 디렉토리: {output_dir}")
    logger.info(f"🖼️ 해상도: {resolution}x{resolution}")
    logger.info(f"🎯 프롬프트: {outpaint_prompt if outpaint_prompt else '기본 프롬프트 사용'}")
    if max_images:
        logger.info(f"🧪 테스트 모드: {max_images}개 이미지 제한")
    logger.info("=" * 50)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="무신사 패션 이미지 Outpainting 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 설정으로 실행
  python outpaint.py --input-dir /path/to/raw_dataset --output-dir /path/to/proc_dataset
  
  # 테스트용 (2개 이미지만)
  python outpaint.py --input-dir /path/to/raw_dataset --output-dir /path/to/proc_dataset --max-images 2
  
  # 커스텀 설정으로 실행
  python outpaint.py \\
    --input-dir /path/to/raw_dataset \\
    --output-dir /path/to/proc_dataset \\
    --resolution 1024 \\
    --outpaint-prompt "a photo of a person (people)" \\
    --num-inference-steps 50 \\
    --guidance-scale 7.5
        """
    )
    
    # 필수 인자
    parser.add_argument(
        "--input-dir", 
        type=str, 
        required=True,
        help="원본 데이터셋 디렉토리 (raw_dataset)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        required=True,
        help="출력 데이터셋 디렉토리 (proc_dataset)"
    )
    
    # 선택적 인자
    parser.add_argument(
        "--images-prefix", 
        type=str, 
        default="images",
        help="이미지 폴더명 (기본값: images)"
    )
    parser.add_argument(
        "--outpaint-prompt", 
        type=str, 
        default="",
        help="확장할 내용 프롬프트 (기본값: 원본 캡션 사용)"
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=1024,
        help="최종 해상도 (기본값: 1024)"
    )
    parser.add_argument(
        "--run-compile", 
        type=arg_as_bool, 
        default=False,
        help="PyTorch 컴파일 사용 여부 (기본값: False)"
    )
    
    # 모델 설정 인자
    parser.add_argument(
        "--num-inference-steps", 
        type=int, 
        default=30,
        help="추론 단계 수 (기본값: 30)"
    )
    parser.add_argument(
        "--guidance-scale", 
        type=float, 
        default=7.5,
        help="프롬프트 준수 강도 (기본값: 7.5)"
    )
    parser.add_argument(
        "--negative-prompt", 
        type=str, 
        default="other people in the background",
        help="제외할 내용 (기본값: other people in the background)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="재현성을 위한 시드 (기본값: 42)"
    )
    
    # 테스트용 인자
    parser.add_argument(
        "--max-images", 
        type=int, 
        default=None,
        help="테스트용 최대 이미지 수 (기본값: 모든 이미지)"
    )
    
    args = parser.parse_args()
    
    # 입력 디렉토리 확인
    if not os.path.exists(args.input_dir):
        logger.error(f"입력 디렉토리가 존재하지 않습니다: {args.input_dir}")
        sys.exit(1)
    
    # 메타데이터 파일 확인
    metadata_path = os.path.join(args.input_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        logger.error(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        sys.exit(1)
    
    logger.info(" 무신사 패션 이미지 Outpainting을 시작합니다...")
    logger.info(f"📁 입력 디렉토리: {args.input_dir}")
    logger.info(f"📁 출력 디렉토리: {args.output_dir}")
    logger.info(f"🖼️ 해상도: {args.resolution}x{args.resolution}")
    logger.info(f"🎯 프롬프트: {args.outpaint_prompt if args.outpaint_prompt else '원본 캡션 사용'}")
    if args.max_images:
        logger.info(f"🧪 테스트 모드: {args.max_images}개 이미지 제한")
    
    # Outpainting 실행
    outpaint_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        images_prefix=args.images_prefix,
        outpaint_prompt=args.outpaint_prompt,
        resolution=args.resolution,
        run_compile=args.run_compile,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
