# MyTorch

> 한국어 API 레퍼런스는 `python scripts/build_docs.py` 실행 후
> [MyTorch 0.8.0 HTML 문서](docs/index.html)에서 확인할 수 있습니다.

NVIDIA GPU에서만 수치 연산을 수행하는 교육용 Tensor 및 자동미분 프레임워크입니다.
현재 버전 0.8.0은 GPU Tensor와 자동미분뿐 아니라 Dataset/DataLoader,
CNN, RNN/LSTM/GRU,
Transformer Encoder, 정규화, LoRA, BitNet, 저정밀 추론과 MoE를 제공합니다.

## 저장소 구조

- `src/mytorch/_tensor`: 생성, 원소별 연산, shape/indexing, 축소, 선형대수 구현
- `src/mytorch/data`: Dataset 구성, GPU batch collation과 DataLoader 구현
- `src/mytorch/nn/functional`: 활성화, 계층 연산, 손실 함수 구현
- `src/mytorch/nn/modules`: 기본 Module부터 CNN, Transformer, BitNet, MoE 계층
- `src/mytorch/optim`: 공통 optimizer 기반과 알고리즘 계열별 구현
- `tests/unit`, `tests/integration`, `tests/gpu`: 단위, 학습 통합, CUDA 시스템 검증

중복된 호환 façade는 제거했으며 공개 API는 `mytorch`, `mytorch.nn`,
`mytorch.data`, `mytorch.nn.functional`, `mytorch.optim`에서 직접 제공합니다.

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

### Dataset과 mini-batch 학습

```python
dataset = mt.data.TensorDataset(x, target)
loader = mt.data.DataLoader(
    dataset, batch_size=64, shuffle=True, seed=2026
)

for epoch in range(100):
    model.train()
    for batch_x, batch_target in loader:
        optimizer.zero_grad()
        loss = F.mse_loss(model(batch_x), batch_target)
        loss.backward()
        optimizer.step()
```

`TensorDataset`은 입력과 라벨처럼 첫 번째 축의 길이가 같은 GPU Tensor를
sample tuple로 연결합니다. 기본 `DataLoader`는 TensorDataset을 한 sample씩
복사하지 않고 batch index를 이용해 GPU에서 한 번에 선택합니다. 일반 사용자
정의 Dataset은 `__len__`과 `__getitem__`을 구현하면 사용할 수 있습니다.

`requires_grad=True`인 실수 Tensor가 연산에 참여하면 동적 계산 그래프가
생성됩니다. `backward()`는 leaf Tensor의 `grad`에 값을 누적하고 기본적으로
사용한 그래프를 해제합니다. 같은 그래프를 다시 사용하려면 첫 호출에
`retain_graph=True`를 지정합니다. 평가처럼 그래프가 필요 없는 코드는
`with mt.no_grad():`로 감쌀 수 있습니다.

완전한 mini-batch 학습과 별도 추론 예제는 다음 순서로 실행합니다.

```powershell
conda activate mytorch-gpu
python examples/train_classifier.py
python examples/infer_classifier.py
```

예제의 `Classifier`는 `nn.Module`을 상속하고 `TensorDataset`과 shuffle이
활성화된 `DataLoader`를 사용해 2차원 점을 left, right, top 세 클래스로
분류합니다.

Tensor 데이터와 연산 결과는 항상 CUDA 장치에 남습니다. 기본 dtype은
`mt.float32`이며 `mt.float16`, `mt.float32`, `mt.float64`를 지원합니다.
현재 공개 연산은 모두 out-of-place이고 자동 CPU 폴백은 없습니다.

### 제공되는 연산

- 생성: `tensor`, `zeros`, `ones`, `full`, `arange`, `linspace`, `eye`,
  `rand`, `randn` 및 `*_like`
- 수학: 사칙연산, 거듭제곱, `exp`, `log`, `log1p`, `logaddexp`,
  `logsumexp`, `sqrt`, `rsqrt`, `round`, `square`, `sin`, `cos`, `sign`,
  `where`, `norm`, `normalize`, `maximum`, `minimum`, `clip`
- 축소: `sum`, `mean`, `prod`, `max`, `min`, `var`, `std`, `argmax`,
  `argmin`
- 행렬: `matmul`, `dot`, `mm`, `bmm`, `outer`
- 형태·선택: `reshape`, `flatten`, `squeeze`, `unsqueeze`, `transpose`,
  `permute`, `expand`, `repeat`, `pad`, `gather`, `scatter_add`, `topk`,
  `masked_fill`, `contiguous`, `cat`, `stack`, `split`, `chunk`
- 활성화: ReLU/ELU/SELU/CELU/PReLU/RReLU 계열, hard·shrink 계열,
  sigmoid/tanh/softmax 계열, GELU/SiLU/Mish 및 GLU/ReGLU/GEGLU/SwiGLU
- 신경망 기반: `Module`, `Parameter`, `ModuleList`, `ParameterList`,
  `Linear`, `Bilinear`, `LazyLinear`, `Sequential`, `Embedding`, `EmbeddingBag`
- 정규화·규제: BatchNorm/InstanceNorm 1D·2D·3D, `LayerNorm`, `RMSNorm`,
  `GroupNorm`, Dropout 1D·2D·3D, `StochasticDepth`
- 공간·순환: Conv/ConvTranspose/MaxPool/AvgPool/AdaptivePool 1D·2D·3D,
  RNN/LSTM/GRU와 대응 Cell
- Transformer: `scaled_dot_product_attention`, `MultiheadAttention`,
  `TransformerEncoderLayer`, `TransformerEncoder`, `RotaryEmbedding`
- 효율 계층: `LowRankLinear`, `LoRALinear`, `SwiGLUFeedForward`,
  `Int8Linear`, `Int4WeightOnlyLinear`, `BitLinear`, `PackedBitLinear`,
  `TopKRouter`, `SparseMoE`
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

## Transformer와 BitLinear

```python
encoder_layer = mt.nn.TransformerEncoderLayer(
    d_model=128,
    nhead=4,
    dim_feedforward=256,
    batch_first=True,
    ffn="swiglu",
)
encoder = mt.nn.TransformerEncoder(encoder_layer, num_layers=2)
encoded = encoder(mt.randn(8, 32, 128))

bit_layer = mt.nn.BitLinear(128, 256)
optimizer = mt.optim.AdamW(bit_layer.parameters(), lr=1e-3)

optimizer.zero_grad()
loss = bit_layer(encoded.detach()).square().mean()
loss.backward()  # STE 기반 ternary QAT
optimizer.step()

bit_layer.eval()
packed = bit_layer.to_inference()
with mt.no_grad():
    result = packed(encoded)  # packed ternary NVRTC 커널
```

`Int8Linear.from_float()`과 `Int4WeightOnlyLinear.from_float()`은 학습된
`Linear`를 추론 전용 계층으로 변환합니다. `PackedBitLinear`도 추론 전용이며
입력이 autograd 그래프를 요구하면 CPU나 일반 Linear로 폴백하지 않고 오류를
발생시킵니다.

## 모델 저장과 복원

```python
mt.save(model.state_dict(), "model.npz")

restored = MyModel()
restored.load_state_dict(mt.load("model.npz", device="cuda:0"))
```

NPZ에는 Parameter와 persistent buffer만 저장됩니다. BatchNorm running 통계,
양자화 scale·packed weight와 LoRA merge 상태도 함께 복원됩니다. 임의 Python
객체를 역직렬화하는 pickle은 사용하지 않습니다.

## API 문서 관리

생성된 문서는 [문서 홈](docs/index.html)과 22개의 모듈·기능별 상세 페이지로 구성됩니다.
설명 원본은 `docs/content`, 공통 디자인은 `docs/assets/css`, 검색·테마·모바일
동작은 `docs/assets/js`에 분리되어 있습니다. HTML은 현재 Python 시그니처를
읽어 생성하므로 코드와 문서의 인자 목록이 어긋나는 것을 줄일 수 있습니다.

```powershell
python scripts/build_docs.py
python scripts/validate_docs.py
python -m http.server 8765 --directory docs
```

첫 명령은 설명 데이터와 실제 공개 API를 합쳐 `docs`에 정적 HTML을 만들고, 두 번째
명령은 버전, API 누락, 페이지·anchor·asset 링크, inline CSS/JavaScript가 없는지
검사합니다. 세 번째 명령으로 로컬 미리보기 서버를 열 수 있습니다.

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
