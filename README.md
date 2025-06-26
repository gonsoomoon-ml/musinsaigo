# Mushinsa Image SDXL LoRA 파인 튜닝


아래는 주요 펄더 입니다. 아래의 순서대로 실행이 필요 하며, 이미지 데이터는 리포에 없습니다. 필요시 저자에게 요청 해주세요. 아래의 원본 코드는 [mushinsago](https://github.com/bits-bytes-nn/musinsaigo) 를 참조 하였습니다.

## 01_setup  
- 01_prepare_data  
    - 데이터 준비를 위한 가상 환경 파일
- 02_train
    - 모델 훈련을 위한 가상 환경 파일
## 02_preproces-image  
- 1_outpaint_image  
- 2_create_person_description  
- train_data
## 03_train  
- sdxl_lora/train.py
## 04_inference
- lora_comparison.ipynb


