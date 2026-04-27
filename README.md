# 한국어 BERT 감성 분석 + Triton 배포 프로젝트

한국어 영화 리뷰 데이터셋(NSMC)을 이용한 감성 분석 모델을 학습하고, ONNX로 변환 후 NVIDIA Triton Inference Server에 배포하는 프로젝트입니다.

\---

## 파일 구조

```
.
├── model.ipynb              # BERT 모델 학습 및 ONNX 변환 노트북
├── config.pbtxt             # Triton Inference Server 모델 설정
└── 이름\_없는\_문서\_1          # Fashion MNIST 분류 예제 (TensorFlow)
```

\---

## 프로젝트 구성

### 1\. 한국어 BERT 감성 분석 (`model.ipynb`)

NSMC(Naver Sentiment Movie Corpus) 데이터셋으로 한국어 BERT 모델(`kykim/bert-kor-base`)을 fine-tuning하여 긍정/부정 감성을 분류합니다.

### 2\. Triton 배포 설정 (`config.pbtxt`)

학습된 모델을 ONNX로 변환하고 NVIDIA Triton Inference Server를 통해 서빙합니다.

### 3\. Fashion MNIST 분류 예제 (`이름\_없는\_문서\_1`)

TensorFlow/Keras를 이용한 이미지 분류 예제입니다.

\---

## 환경 요구사항

### 하드웨어

* CUDA 지원 GPU (권장: CUDA 11.x 이상, VRAM 8GB 이상)
* Triton 서버용 CPU 또는 GPU

### 소프트웨어

* Python 3.8+
* CUDA Toolkit 11.x 이상

\---

## 설치 방법

### 1\. 가상환경 생성 (권장)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\\Scripts\\activate         # Windows
```

### 2\. 필수 패키지 설치

```bash
# PyTorch (CUDA 11.8 기준)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 주요 라이브러리
pip install transformers
pip install Korpora
pip install pandas
pip install tqdm
pip install onnx
pip install mlflow
pip install jupyter

# TensorFlow (Fashion MNIST 예제용)
pip install tensorflow matplotlib
```

또는 한 번에 설치:

```bash
pip install torch transformers Korpora pandas tqdm onnx mlflow jupyter tensorflow matplotlib
```

\---

## 실행 방법

### 1단계: BERT 감성 분석 모델 학습

```bash
jupyter notebook model.ipynb
```

노트북을 열고 셀을 순서대로 실행합니다.

**주요 실행 흐름:**

|단계|내용|
|-|-|
|데이터 로드|NSMC 데이터셋 로드 및 전처리|
|토크나이저|`kykim/bert-kor-base` 토크나이저 적용|
|모델 정의|`CustomBertModel` (BERT + Dropout + FC)|
|학습|100 Epochs, Adam optimizer (lr=1e-5)|
|ONNX 변환|`torch.onnx.export`로 모델 변환|
|MLflow 등록|모델을 MLflow에 로깅|

**학습 파라미터:**

```python
CHECKPOINT\_NAME = 'kykim/bert-kor-base'
batch\_size = 8
learning\_rate = 1e-5
num\_epochs = 100
dropout\_rate = 0.5
```

**ONNX 모델 저장 경로:**

```
/home/eternal/model\_repository/test/1/bert\_model.onnx
```

> 필요에 따라 경로를 수정하세요.

\---

### 2단계: Triton Inference Server 배포

#### 모델 리포지토리 구조 설정

```
model\_repository/
└── bert\_kor\_base1/
    ├── config.pbtxt
    └── 1/
        └── bert\_model.onnx
```

```bash
mkdir -p model\_repository/bert\_kor\_base1/1
cp bert\_model.onnx model\_repository/bert\_kor\_base1/1/
cp config.pbtxt model\_repository/bert\_kor\_base1/
```

#### Triton 서버 실행 (Docker)

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v $(pwd)/model\_repository:/models \\
  nvcr.io/nvidia/tritonserver:23.10-py3 \\
  tritonserver --model-repository=/models
```

GPU 사용 시:

```bash
docker run --rm --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v $(pwd)/model\_repository:/models \\
  nvcr.io/nvidia/tritonserver:23.10-py3 \\
  tritonserver --model-repository=/models
```

#### 서버 상태 확인

```bash
curl http://localhost:8000/v2/health/ready
```

\---

#### Triton 클라이언트 추론 예시

```python
import tritonclient.http as httpclient
import numpy as np
from transformers import BertTokenizerFast

# 클라이언트 연결
client = httpclient.InferenceServerClient(url="localhost:8000")

# 입력 문장 토크나이징
tokenizer = BertTokenizerFast.from\_pretrained("kykim/bert-kor-base")
sentence = "이 영화 정말 재미있어요!"

tokens = tokenizer(
    sentence,
    return\_tensors="np",
    truncation=True,
    padding="max\_length",
    max\_length=512,
    add\_special\_tokens=True
)

# 입력 데이터 준비
input\_ids = tokens\["input\_ids"].astype(np.int64)
attention\_mask = tokens\["attention\_mask"].astype(np.int64)
token\_type\_ids = np.zeros\_like(attention\_mask, dtype=np.int64)

# Triton 입력 설정
inputs = \[
    httpclient.InferInput("input\_ids", input\_ids.shape, "INT64"),
    httpclient.InferInput("attention\_mask", attention\_mask.shape, "INT64"),
    httpclient.InferInput("token\_type\_ids", token\_type\_ids.shape, "INT64"),
]
inputs\[0].set\_data\_from\_numpy(input\_ids)
inputs\[1].set\_data\_from\_numpy(attention\_mask)
inputs\[2].set\_data\_from\_numpy(token\_type\_ids)

outputs = \[httpclient.InferRequestedOutput("output")]

# 추론 실행
response = client.infer("bert\_kor\_base1", inputs=inputs, outputs=outputs)
result = response.as\_numpy("output")
label = np.argmax(result, axis=1)\[0]
print("긍정" if label == 1 else "부정")
```

\---

### 3단계: Fashion MNIST 예제 실행

```bash
python 이름\_없는\_문서\_1.py
```

또는 Python 인터프리터에서 실행:

```python
exec(open('이름\_없는\_문서\_1').read())
```

\---

## 모델 설명

### `config.pbtxt` 설정 요약

|항목|값|
|-|-|
|모델 이름|`bert\_kor\_base1`|
|플랫폼|`onnxruntime\_onnx`|
|최대 배치 크기|동적 (`max\_batch\_size: 0`)|
|입력|`input\_ids`, `attention\_mask`, `token\_type\_ids` (INT64, shape: \[-1, 512])|
|출력|`output` (FP32, shape: \[-1, 2])|
|실행 환경|CPU (`KIND\_CPU`)|

GPU 배포 시 `config.pbtxt`에서 `KIND\_CPU` → `KIND\_GPU`로 변경하세요.

## 데이터셋

NSMC(Naver Sentiment Movie Corpus)는 Korpora 라이브러리를 통해 자동 다운로드됩니다.

```python
from Korpora import Korpora
corpus = Korpora.load("nsmc")
```

데이터는 `\~/Korpora/nsmc/` 경로에 저장됩니다.

|구분|경로|
|-|-|
|학습 데이터|`\~/Korpora/nsmc/ratings\_train.txt`|
|테스트 데이터|`\~/Korpora/nsmc/ratings\_test.txt`|

> 노트북에서는 학습 데이터 1,000개, 테스트 데이터 500개를 샘플링하여 사용합니다.

\---

## 주의사항

* GPU 인덱스는 코드에서 `cuda:1`로 설정되어 있습니다. 단일 GPU 환경이라면 `cuda:0`으로 변경하세요.
* ONNX 모델 저장 경로(`/home/eternal/...`)는 실제 환경에 맞게 수정이 필요합니다.
* Triton 서버의 `tritonclient` 설치: `pip install tritonclient\[http]`

\---

## 의존성 전체 목록

```
torch
torchvision
torchaudio
transformers
Korpora
pandas
tqdm
onnx
mlflow
jupyter
tensorflow
matplotlib
tritonclient\[http]
```

\---

## 📝 라이선스

이 프로젝트는 개인/교육 목적으로 작성되었습니다.  
사용된 사전학습 모델 `kykim/bert-kor-base`는 해당 모델의 라이선스를 따릅니다.

