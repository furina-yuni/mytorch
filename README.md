# MyTorch

NVIDIA GPU에서만 수치 연산을 수행하는 교육용 Tensor 및 자동미분 프레임워크입니다.
현재 단계에서는 GPU Tensor, 역방향 자동미분, MLP 레이어, 다양한 손실 함수와
13종의 dense optimizer를 제공합니다.

## 빠른 예제

```python
import mytorch as mt
from mytorch.nn import functional as F

x = mt.tensor([[1.0, 2.0], [3.0, 4.0]])
w = mt.randn(2, 3)

logits = x @ w
probabilities = F.softmax(logits, dim=-1)

print(logits)
print(probabilities.numpy())  # 명시적으로 CPU NumPy 배열로 복사
```

## 자동미분과 학습

```python
import mytorch as mt
from mytorch.nn import functional as F

x = mt.tensor([[1.0, 2.0], [3.0, 4.0]])
target = mt.tensor([[3.0], [7.0]])

model = mt.nn.Sequential(
    mt.nn.Linear(2, 16),
    mt.nn.ReLU(),
    mt.nn.Linear(16, 1),
)
optimizer = mt.optim.SGD(model.parameters(), lr=0.05)

optimizer.zero_grad()
prediction = model(x)  # 순전파와 동적 계산 그래프 생성
loss = F.mse_loss(prediction, target)
loss.backward()  # GPU에서 역전파
optimizer.step()  # GPU 파라미터 갱신
```

`requires_grad=True`인 실수 Tensor가 연산에 참여하면 동적 계산 그래프가
생성됩니다. `backward()`는 leaf Tensor의 `grad`에 값을 누적하고 기본적으로
사용한 그래프를 해제합니다. 같은 그래프를 다시 사용하려면 첫 호출에
`retain_graph=True`를 지정합니다. 평가처럼 그래프가 필요 없는 코드는
`with mt.no_grad():`로 감쌀 수 있습니다.

완전한 신경망 클래스, 합성 데이터 학습과 새 데이터 예측 예제는 다음 명령으로
실행합니다.

```powershell
conda activate mytorch-gpu
python examples/simple_classifier.py
```

예제의 `SimpleClassifier`는 `nn.Module`을 상속하고 두 개의 `Linear` 레이어와
`ReLU`를 사용해 2차원 점을 left, right, top 세 클래스로 분류합니다.

Tensor 데이터와 연산 결과는 항상 CUDA 장치에 남습니다. 기본 dtype은
`mt.float32`이며 `mt.float16`, `mt.float32`, `mt.float64`를 지원합니다.
현재 공개 연산은 모두 out-of-place이고 자동 CPU 폴백은 없습니다.

### 제공되는 연산

- 생성: `tensor`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`,
  `rand`, `randn` 및 `*_like`
- 수학: 사칙연산, 거듭제곱, `exp`, `log`, `log1p`, `logaddexp`,
  `logsumexp`, `sqrt`, `square`, `sin`, `cos`, `sign`, `where`, `norm`,
  `normalize`, `maximum`, `minimum`, `clip`
- 축소: `sum`, `mean`, `prod`, `max`, `min`, `var`, `std`, `argmax`,
  `argmin`
- 행렬: `matmul`, `dot`, `mm`, `bmm`, `outer`
- 형태: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`, `permute`,
  `cat`, `stack`, `split`, `chunk` 및 읽기 전용 인덱싱
- 활성화: ReLU/ELU/SELU/CELU/PReLU/RReLU 계열, hard·shrink 계열,
  sigmoid/tanh/softmax 계열, GELU/SiLU/Mish 및 GLU/ReGLU/GEGLU/SwiGLU
- 신경망: `Module`, `Parameter`, `Linear`, `Sequential`, `Flatten`, 활성화 레이어
- 손실: MSE/L1/Huber 계열, CrossEntropy/NLL/BCE/KL 계열, margin·triplet·
  cosine·contrastive 계열, Focal 및 Dice
- 최적화: `SGD`, `Adagrad`, `RMSprop`, `Adadelta`, `Adam`, `AdamW`,
  `Adamax`, `NAdam`, `RAdam`, `ASGD`, `Rprop`, `Adafactor`, `Lion`

활성화 함수는 `mytorch.nn.functional`에서 사용합니다. `relu`, `sigmoid`,
`tanh`, `softmax`, `log_softmax`는 Tensor 메서드로도 호출할 수 있습니다.

## 선택 가이드

- 일반적인 첫 선택은 `AdamW`입니다. 가중치 감쇠를 moment 계산과 분리합니다.
- 작은 MLP나 고전적인 실험에는 `SGD(momentum=0.9)`가 단순하고 해석하기 쉽습니다.
- 큰 2차원 가중치의 optimizer 메모리를 줄이려면 factor state를 쓰는
  `Adafactor`를 고려합니다.
- `Lion`은 parameter마다 momentum 배열 하나만 저장하는 sign 기반 방식입니다.
- 불균형 이진 분류에는 `BCEWithLogitsLoss` 또는 `FocalLoss`, binary mask에는
  `DiceLoss`를 사용할 수 있습니다. 확률을 먼저 sigmoid로 만들기보다 logits를
  직접 받는 손실이 극단값에서 더 안정적입니다.

```python
model = mt.nn.Sequential(
    mt.nn.Linear(32, 64),
    mt.nn.GELU(),
    mt.nn.Linear(64, 4),
)
optimizer = mt.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
criterion = mt.nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer.zero_grad()
loss = criterion(model(features), labels)
loss.backward()
optimizer.step()
```

Optimizer는 parameter-group별 옵션도 지원합니다.

```python
optimizer = mt.optim.AdamW(
    [
        {"params": [model[0].weight], "lr": 1e-3},
        {"params": [model[0].bias], "lr": 2e-3, "weight_decay": 0.0},
    ],
    weight_decay=1e-2,
)
```

## 환경 만들기

Miniconda가 PowerShell에 아직 연결되지 않았다면 한 번만 실행합니다.

```powershell
& 'C:\Users\mintc\miniconda3\Scripts\conda.exe' init powershell
```

PowerShell을 다시 연 다음 프로젝트 루트에서 실행합니다.

```powershell
conda create -n mytorch-gpu --override-channels -c conda-forge `
  --strict-channel-priority -f environment.yml
conda activate mytorch-gpu
python -m pip install -e . --no-deps --no-build-isolation
```

`--override-channels`는 로컬 conda 설정의 `defaults` 채널을 완전히 제외합니다.

CUDA와 외부 Python 의존성은 모두 `mytorch-gpu` 환경의 conda-forge 패키지로
관리합니다. 시스템에는 NVIDIA 디스플레이 드라이버만 필요합니다.

## 확인하기

```powershell
conda activate mytorch-gpu
python scripts/gpu_smoke_test.py
pytest
ruff check .
ruff format --check .
```

`nvcc`는 현재 환경의 필수 항목이 아닙니다. 이후 `.cu` 파일의 오프라인 컴파일이
필요해질 때 `cuda-nvcc=13.2`를 같은 conda 환경에 추가합니다.

## 정확한 환경 복원

`environment.yml`은 사람이 관리하는 사양이며, `conda-win-64.lock`은 현재 Windows
환경에서 검증된 패키지 URL을 고정합니다.

```powershell
conda create -n mytorch-gpu --override-channels -c conda-forge `
  --file conda-win-64.lock
conda activate mytorch-gpu
python -m pip install -e . --no-deps --no-build-isolation
```
