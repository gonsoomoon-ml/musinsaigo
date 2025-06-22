# Image Description Generator using Amazon Bedrock Claude 3.7

이 스크립트는 Amazon Bedrock의 Claude 3.7 Sonnet 모델을 사용하여 패션 이미지에 대한 상세한 인물 중심 설명을 생성합니다.

## 기능

- Amazon Bedrock Claude 3.7 Sonnet 모델을 사용한 이미지 분석
- 인물 중심의 상세한 패션 설명 생성 (50개 단어 내외)
- 기존 metadata.jsonl 파일의 텍스트 필드 업데이트
- 로깅 및 에러 처리
- Rate limiting 방지를 위한 지연 처리

## 사전 요구사항

### 1. AWS 설정
```bash
# AWS CLI 설치 및 설정
aws configure
```

### 2. Amazon Bedrock 접근 권한
- Amazon Bedrock 서비스에 대한 접근 권한이 필요합니다
- Claude 3.7 Sonnet 모델 사용 권한이 필요합니다

### 3. Python 의존성
```bash
pip install -r requirements.txt
```

## 사용법

### 1. 스크립트 실행
```bash
# 실행 권한 부여
chmod +x run_generator.sh

# 스크립트 실행
./run_generator.sh
```

### 2. 직접 Python 실행
```bash
python generate_descriptions.py
```

## 설정

### 지역 설정
`generate_descriptions.py` 파일에서 AWS 지역을 변경할 수 있습니다:

```python
region_name = "us-east-1"  # 사용 가능한 Bedrock 지역으로 변경
```

### 입력/출력 파일 경로
기본 설정:
- 입력: `/home/ubuntu/musinsaigo/test_data/metadata.jsonl`
- 출력: `/home/ubuntu/musinsaigo/test_data/metadata_updated.jsonl`

## 출력 형식

생성된 설명은 다음과 같은 형식으로 저장됩니다:

```json
{"file_name": "images/example.jpg", "text": "A young Korean woman with shoulder-length dark hair wearing a casual white t-shirt and light blue jeans, standing confidently in an office environment with natural lighting, showcasing everyday street style fashion"}
```

## 로그 파일

실행 과정은 `description_generation.log` 파일에 기록됩니다:
- 처리 진행 상황
- 에러 메시지
- 생성된 설명 미리보기

## 에러 처리

- 이미지 파일이 없는 경우: 원본 텍스트 유지
- API 호출 실패: 에러 로그 기록 후 다음 이미지 처리
- 네트워크 오류: 재시도 로직 포함

## 비용 고려사항

Amazon Bedrock 사용 시 다음 비용이 발생합니다:
- Claude 3.7 Sonnet: 입력 토큰당 요금
- 이미지 분석: 이미지당 추가 요금

## 주의사항

1. **Rate Limiting**: API 호출 간 1초 지연을 두어 rate limiting을 방지합니다
2. **이미지 크기**: 큰 이미지 파일은 처리 시간이 오래 걸릴 수 있습니다
3. **네트워크**: 안정적인 인터넷 연결이 필요합니다
4. **AWS 크레딧**: 충분한 AWS 크레딧이 필요합니다

## 문제 해결

### AWS 인증 오류
```bash
aws configure
aws sts get-caller-identity
```

### Bedrock 접근 권한 오류
AWS IAM에서 Bedrock 권한을 확인하세요:
- `bedrock:InvokeModel`
- `bedrock:InvokeModelWithResponseStream`

### 이미지 파일 오류
이미지 파일 경로와 권한을 확인하세요:
```bash
ls -la /home/ubuntu/musinsaigo/test_data/images/
``` 