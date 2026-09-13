# ruff: noqa: E501
"""Build the multi-page MyTorch HTML API reference from live Python signatures."""

from __future__ import annotations

import html
import importlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mytorch as mt

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
CONTENT = DOCS_ROOT / "content"
API_DIR = DOCS_ROOT / "api"


@dataclass(frozen=True)
class ApiItem:
    name: str
    qualified: str
    value: Any
    kind: str
    owner: str | None = None


SUMMARY = {
    "Tensor": "CUDA 장치의 다차원 배열과 자동미분 상태를 함께 보관하는 핵심 자료형입니다.",
    "Parameter": "Module이 학습 대상으로 자동 등록하는 requires_grad=True leaf Tensor입니다.",
    "Module": "신경망 계층, Parameter, buffer, 학습·평가 모드를 재귀적으로 관리하는 기반 클래스입니다.",
    "Linear": "입력 마지막 축에 affine 변환 y = xWᵀ + b를 적용합니다.",
    "Bilinear": "두 입력의 마지막 특성축 사이에 학습 가능한 bilinear 변환을 적용합니다.",
    "LazyLinear": "첫 forward의 입력 마지막 차원으로 in_features를 결정해 Linear 파라미터를 지연 생성합니다.",
    "Embedding": "정수 token index를 학습 가능한 dense embedding vector로 조회합니다.",
    "EmbeddingBag": "여러 embedding을 bag 단위로 sum, mean 또는 max 집계합니다.",
    "Sequential": "등록 순서대로 자식 Module의 출력을 다음 Module 입력으로 전달합니다.",
    "ModuleList": "Module iterable을 등록 상태를 유지한 채 list처럼 보관합니다.",
    "ParameterList": "Parameter iterable을 등록 상태를 유지한 채 list처럼 보관합니다.",
    "Identity": "입력을 변경하지 않고 그대로 반환합니다.",
    "Flatten": "지정한 연속 축 범위를 하나의 축으로 평탄화합니다.",
    "MultiheadAttention": "query, key, value projection과 multi-head scaled dot-product attention을 수행합니다.",
    "TransformerEncoderLayer": "self-attention, residual, normalization, feed-forward 블록으로 encoder 한 층을 구성합니다.",
    "TransformerEncoder": "동일 구조의 TransformerEncoderLayer를 여러 층 순차 적용합니다.",
    "RotaryEmbedding": "위치별 sin·cos 회전값을 GPU cache로 만들어 rotary position embedding에 사용합니다.",
    "LowRankLinear": "두 개의 작은 행렬 곱으로 Linear weight를 저랭크 분해해 파라미터 수를 줄입니다.",
    "LoRALinear": "고정 가능한 base Linear에 학습 가능한 저랭크 adapter를 더합니다.",
    "SwiGLUFeedForward": "SwiGLU gate를 사용하는 Transformer형 feed-forward network입니다.",
    "Int8Linear": "per-token activation과 per-output-channel weight 양자화를 사용하는 추론 전용 Linear입니다.",
    "Int4WeightOnlyLinear": "groupwise 4-bit packed weight를 사용하는 추론 전용 Linear입니다.",
    "BitLinear": "ternary weight와 INT8 activation의 straight-through estimator로 학습하는 BitNet 계층입니다.",
    "PackedBitLinear": "ternary weight를 2-bit로 압축하고 NVRTC kernel에서 직접 계산하는 추론 계층입니다.",
    "TopKRouter": "각 token에 대한 expert score를 계산하고 확률이 높은 k개 expert를 선택합니다.",
    "SparseMoE": "선택된 expert만 실행하고 routing 가중합과 load-balance loss를 반환합니다.",
    "Optimizer": "dense GPU Parameter 그룹과 optimizer state의 공통 생명주기를 관리합니다.",
    "save": "Tensor state mapping을 NPZ 배열과 JSON 메타데이터로 저장합니다.",
    "load": "저장된 NPZ state mapping을 지정 CUDA 장치의 Tensor로 복원합니다.",
    "is_available": "CUDA 런타임을 초기화할 수 있고 접근 가능한 GPU가 있는지 확인합니다.",
    "device_count": "현재 CUDA 런타임이 인식하는 GPU 장치 개수를 반환합니다.",
    "no_grad": "블록 안에서 자동미분 그래프 기록을 일시적으로 끄는 context manager입니다.",
    "enable_grad": "no_grad 안에서도 그래프 기록을 일시적으로 다시 켜는 context manager입니다.",
    "is_grad_enabled": "현재 실행 문맥에서 자동미분 기록이 활성화되었는지 반환합니다.",
    "set_grad_enabled": "자동미분 기록 상태를 지정 값으로 잠시 바꾸는 내부 친화적 context manager입니다.",
    "manual_seed": "현재 사용 가능한 모든 CuPy GPU 난수 생성기의 seed를 고정합니다.",
    "tensor": "Python·NumPy·CuPy 데이터를 지정 CUDA 장치의 Tensor로 변환합니다.",
    "arange": "일정 간격의 1차원 값을 GPU Tensor로 생성합니다.",
    "linspace": "시작과 끝을 포함한 균등 간격 값을 GPU Tensor로 생성합니다.",
    "eye": "주대각선이 1인 2차원 GPU Tensor를 생성합니다.",
    "rand": "[0, 1) 균등분포에서 GPU 난수를 생성합니다.",
    "randn": "표준정규분포에서 GPU 난수를 생성합니다.",
    "where": "boolean condition에 따라 input 또는 other의 원소를 선택합니다.",
    "gather": "index Tensor가 가리키는 값을 지정 축에서 모읍니다.",
    "scatter_add": "source 값을 index가 가리키는 출력 위치에 GPU에서 누적합니다.",
    "topk": "지정 축에서 가장 크거나 작은 k개 값과 index를 반환합니다.",
    "pad": "Tensor의 마지막 축부터 지정한 폭만큼 상수 padding을 추가합니다.",
    "cat": "기존 축을 따라 여러 Tensor를 이어 붙입니다.",
    "stack": "새 축을 만들고 여러 Tensor를 그 축을 따라 쌓습니다.",
    "split": "Tensor를 지정 축에서 크기 또는 section 목록에 따라 나눕니다.",
    "chunk": "Tensor를 지정 축에서 가능한 한 균등한 조각으로 나눕니다.",
    "matmul": "벡터, 행렬, batched N-D 입력에 PyTorch형 행렬곱 규칙을 적용합니다.",
    "dot": "정확히 두 1차원 Tensor의 내적을 계산합니다.",
    "mm": "정확히 두 2차원 Tensor의 행렬곱을 계산합니다.",
    "bmm": "동일 batch 크기를 가진 두 3차원 Tensor의 batch 행렬곱을 계산합니다.",
    "outer": "두 1차원 Tensor의 외적 행렬을 계산합니다.",
    "scaled_dot_product_attention": "안정적인 softmax, mask, causal, dropout을 포함한 attention 핵심 연산입니다.",
    "apply_rotary_pos_emb": "query와 key의 짝수·홀수 feature 쌍을 sin·cos로 회전합니다.",
}


ACTIVATIONS = {
    "relu": "음수는 0, 양수는 그대로 두는 ReLU 활성화를 적용합니다.",
    "leaky_relu": "음수 구간에 작은 고정 기울기를 두는 Leaky ReLU를 적용합니다.",
    "sigmoid": "입력을 0과 1 사이로 변환하는 수치 안정적 sigmoid를 적용합니다.",
    "tanh": "입력을 -1과 1 사이로 변환하는 hyperbolic tangent를 적용합니다.",
    "softmax": "지정 축의 값을 합이 1인 확률로 정규화합니다.",
    "log_softmax": "log-sum-exp 안정화로 log-softmax를 계산합니다.",
    "gelu": "Gaussian Error Linear Unit을 정확식 또는 tanh 근사식으로 계산합니다.",
    "silu": "x·sigmoid(x)인 SiLU/Swish 활성화를 적용합니다.",
    "softplus": "ReLU의 매끄러운 근사 log(1 + exp(βx))/β를 안정적으로 계산합니다.",
    "threshold": "입력이 threshold 이하이면 지정 value로 바꿉니다.",
    "hardtanh": "값을 min_val과 max_val 구간으로 제한합니다.",
    "relu6": "ReLU 결과를 최대 6으로 제한합니다.",
    "elu": "음수 구간에 alpha·(exp(x)-1)을 사용하는 ELU를 적용합니다.",
    "selu": "self-normalizing network용 고정 scale과 alpha의 SELU를 적용합니다.",
    "celu": "alpha로 연속 미분 가능하게 조정한 CELU를 적용합니다.",
    "prelu": "학습 가능한 음수 기울기를 scalar 또는 channel별로 적용합니다.",
    "rrelu": "학습 중 무작위, 평가 중 평균 음수 기울기를 사용하는 RReLU를 적용합니다.",
    "hardsigmoid": "구간별 선형 sigmoid 근사를 계산합니다.",
    "hardswish": "x·hardsigmoid(x)인 모바일 친화적 활성화를 계산합니다.",
    "hardshrink": "절댓값이 lambd 이하인 원소를 0으로 만듭니다.",
    "softshrink": "임계값 바깥 값을 0 방향으로 lambd만큼 이동합니다.",
    "tanhshrink": "x - tanh(x)를 계산합니다.",
    "logsigmoid": "log(sigmoid(x))를 극단 입력에서도 안정적으로 계산합니다.",
    "softsign": "x / (1 + |x|)를 계산합니다.",
    "softmin": "-input에 softmax를 적용해 작은 값에 큰 확률을 부여합니다.",
    "mish": "x·tanh(softplus(x))인 Mish 활성화를 적용합니다.",
    "glu": "입력을 반으로 나눠 첫 절반에 sigmoid gate를 곱합니다.",
    "reglu": "입력을 반으로 나눠 첫 절반에 ReLU gate를 곱합니다.",
    "geglu": "입력을 반으로 나눠 첫 절반에 GELU gate를 곱합니다.",
    "swiglu": "입력을 반으로 나눠 첫 절반에 SiLU gate를 곱합니다.",
}


LOSSES = {
    "mse_loss": "prediction과 target의 제곱 오차를 계산합니다.",
    "l1_loss": "prediction과 target의 절댓값 오차를 계산합니다.",
    "smooth_l1_loss": "작은 오차는 quadratic, 큰 오차는 linear인 Smooth L1을 계산합니다.",
    "huber_loss": "delta 경계로 quadratic과 linear 구간을 연결한 Huber loss를 계산합니다.",
    "cross_entropy": "logits에서 안정적 log-softmax와 class-index NLL을 한 번에 계산합니다.",
    "nll_loss": "log-probability와 class index target의 negative log likelihood를 계산합니다.",
    "binary_cross_entropy": "확률 입력과 binary target 사이의 cross entropy를 계산합니다.",
    "binary_cross_entropy_with_logits": "sigmoid와 binary cross entropy를 수치 안정적인 한 식으로 계산합니다.",
    "kl_div": "입력 log-probability와 target 분포 사이의 KL divergence 항을 계산합니다.",
    "poisson_nll_loss": "Poisson 분포를 가정한 count target의 negative log likelihood를 계산합니다.",
    "gaussian_nll_loss": "평균 prediction과 variance를 이용해 Gaussian negative log likelihood를 계산합니다.",
    "hinge_embedding_loss": "유사·비유사 label에 따라 거리 입력에 hinge penalty를 적용합니다.",
    "margin_ranking_loss": "두 입력의 순서가 target 방향으로 margin만큼 벌어지도록 학습합니다.",
    "soft_margin_loss": "binary target에 부드러운 logistic margin loss를 계산합니다.",
    "multilabel_soft_margin_loss": "class별 binary logistic loss를 계산해 multilabel 예측을 학습합니다.",
    "cosine_embedding_loss": "두 embedding의 cosine similarity를 target에 따라 가깝거나 멀게 만듭니다.",
    "triplet_margin_loss": "anchor-positive 거리가 anchor-negative보다 margin만큼 작아지도록 학습합니다.",
    "multi_margin_loss": "정답 class score가 다른 class보다 margin만큼 높아지도록 hinge loss를 계산합니다.",
    "sigmoid_focal_loss": "쉬운 binary 예제의 기여를 줄이고 어려운 예제에 집중하는 focal loss입니다.",
    "dice_loss": "예측 mask와 target mask의 soft Dice overlap을 최대화하는 손실입니다.",
    "contrastive_loss": "유사 쌍은 가깝게, 비유사 쌍은 margin 밖으로 보내는 metric loss입니다.",
}


OPTIMIZERS = {
    "SGD": "gradient, momentum, Nesterov와 선택적 L2 감쇠로 Parameter를 갱신합니다.",
    "Adam": "1차·2차 moment의 지수 이동 평균과 bias correction으로 Parameter를 갱신합니다.",
    "AdamW": "Adam moment와 분리된 decoupled weight decay를 적용합니다.",
    "Adamax": "무한 노름 기반 second moment를 사용하는 Adam 변형입니다.",
    "NAdam": "Adam의 adaptive moment에 Nesterov momentum을 결합합니다.",
    "RAdam": "학습 초기 adaptive learning rate의 분산을 rectification하는 Adam 변형입니다.",
    "Adagrad": "누적 제곱 gradient로 자주 갱신되는 파라미터의 유효 학습률을 낮춥니다.",
    "RMSprop": "제곱 gradient 이동 평균으로 파라미터별 learning rate를 조정합니다.",
    "Adadelta": "gradient와 update의 제곱 이동 평균 비율로 갱신 scale을 정합니다.",
    "ASGD": "감쇠 학습률과 일정 시점 이후의 parameter averaging을 사용하는 SGD입니다.",
    "Rprop": "gradient 부호 변화에 따라 원소별 step size를 늘리거나 줄입니다.",
    "Adafactor": "행렬 second moment를 행·열로 factorize해 optimizer state 메모리를 줄입니다.",
    "Lion": "하나의 momentum state와 sign update를 사용하는 메모리 효율 optimizer입니다.",
}


SIGNATURE_OVERRIDES = {
    "EmbeddingBag": "(num_embeddings, embedding_dim, padding_idx=None, *, mode='mean', device='cuda:0', dtype=float32)",
    "RNNCell": "(input_size, hidden_size, bias=True, *, nonlinearity='tanh', device='cuda:0', dtype=float32)",
    "RNN": "(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, *, nonlinearity='tanh', device='cuda:0', dtype=float32)",
}


CLASS_TO_FUNCTION = {
    "ReLU": "relu",
    "LeakyReLU": "leaky_relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "LogSoftmax": "log_softmax",
    "GELU": "gelu",
    "SiLU": "silu",
    "Softplus": "softplus",
    "Threshold": "threshold",
    "Hardtanh": "hardtanh",
    "ReLU6": "relu6",
    "ELU": "elu",
    "SELU": "selu",
    "CELU": "celu",
    "PReLU": "prelu",
    "RReLU": "rrelu",
    "Hardsigmoid": "hardsigmoid",
    "Hardswish": "hardswish",
    "Hardshrink": "hardshrink",
    "Softshrink": "softshrink",
    "Tanhshrink": "tanhshrink",
    "LogSigmoid": "logsigmoid",
    "Softsign": "softsign",
    "Softmin": "softmin",
    "Mish": "mish",
    "GLU": "glu",
    "ReGLU": "reglu",
    "GEGLU": "geglu",
    "SwiGLU": "swiglu",
    "MSELoss": "mse_loss",
    "L1Loss": "l1_loss",
    "SmoothL1Loss": "smooth_l1_loss",
    "HuberLoss": "huber_loss",
    "CrossEntropyLoss": "cross_entropy",
    "NLLLoss": "nll_loss",
    "BCELoss": "binary_cross_entropy",
    "BCEWithLogitsLoss": "binary_cross_entropy_with_logits",
    "KLDivLoss": "kl_div",
    "PoissonNLLLoss": "poisson_nll_loss",
    "GaussianNLLLoss": "gaussian_nll_loss",
    "HingeEmbeddingLoss": "hinge_embedding_loss",
    "MarginRankingLoss": "margin_ranking_loss",
    "SoftMarginLoss": "soft_margin_loss",
    "MultiLabelSoftMarginLoss": "multilabel_soft_margin_loss",
    "CosineEmbeddingLoss": "cosine_embedding_loss",
    "TripletMarginLoss": "triplet_margin_loss",
    "MultiMarginLoss": "multi_margin_loss",
    "FocalLoss": "sigmoid_focal_loss",
    "DiceLoss": "dice_loss",
    "ContrastiveLoss": "contrastive_loss",
}


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _signature(item: ApiItem) -> str:
    if item.name in SIGNATURE_OVERRIDES and item.owner is None:
        return SIGNATURE_OVERRIDES[item.name]
    try:
        signature = str(inspect.signature(item.value))
    except (TypeError, ValueError):
        return ""
    signature = signature.replace("<class 'cupy.float32'>", "float32")
    signature = signature.replace("<class 'cupy.float64'>", "float64")
    return signature


def _parameters(item: ApiItem) -> list[inspect.Parameter]:
    if item.owner is None and item.name in SIGNATURE_OVERRIDES:
        positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
        keyword = inspect.Parameter.KEYWORD_ONLY
        common_device = [
            inspect.Parameter("device", keyword, default="cuda:0"),
            inspect.Parameter("dtype", keyword, default=mt.float32),
        ]
        if item.name == "EmbeddingBag":
            return [
                inspect.Parameter("num_embeddings", positional, annotation=int),
                inspect.Parameter("embedding_dim", positional, annotation=int),
                inspect.Parameter("padding_idx", positional, default=None),
                inspect.Parameter("mode", keyword, default="mean"),
                *common_device,
            ]
        base = [
            inspect.Parameter("input_size", positional, annotation=int),
            inspect.Parameter("hidden_size", positional, annotation=int),
        ]
        if item.name == "RNNCell":
            return [
                *base,
                inspect.Parameter("bias", positional, default=True),
                inspect.Parameter("nonlinearity", keyword, default="tanh"),
                *common_device,
            ]
        return [
            *base,
            inspect.Parameter("num_layers", positional, default=1),
            inspect.Parameter("bias", positional, default=True),
            inspect.Parameter("batch_first", positional, default=False),
            inspect.Parameter("dropout", positional, default=0.0),
            inspect.Parameter("bidirectional", positional, default=False),
            inspect.Parameter("nonlinearity", keyword, default="tanh"),
            *common_device,
        ]
    try:
        signature = inspect.signature(item.value)
    except (TypeError, ValueError):
        return []
    return [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name not in {"self", "cls"}
    ]


def _summary(item: ApiItem) -> str:
    bare = item.name.split(".")[-1]
    snake = CLASS_TO_FUNCTION.get(bare, bare)
    if bare in SUMMARY:
        return SUMMARY[bare]
    if snake in ACTIVATIONS:
        return ACTIVATIONS[snake]
    if snake in LOSSES:
        return LOSSES[snake]
    if bare in OPTIMIZERS:
        return OPTIMIZERS[bare]
    if bare.startswith("ConvTranspose"):
        return (
            "학습 가능한 kernel로 입력의 공간 크기를 확장하는 전치 합성곱 계층입니다."
        )
    if bare.startswith("Conv"):
        return (
            "GPU im2col과 matrix multiplication으로 N차원 합성곱을 수행하는 계층입니다."
        )
    if "Pool" in bare:
        action = "최댓값" if "Max" in bare else "평균값"
        return f"공간 window 또는 목표 출력 구간에서 {action}을 집계하는 pooling 계층입니다."
    if bare.startswith("BatchNorm"):
        return (
            "mini-batch와 channel별 통계로 정규화하고 평가용 running 통계를 관리합니다."
        )
    if bare.startswith("InstanceNorm"):
        return "sample과 channel별 공간 통계로 입력을 정규화합니다."
    if bare == "LayerNorm":
        return "각 sample의 마지막 연속 feature 축을 평균과 분산으로 정규화합니다."
    if bare == "RMSNorm":
        return "평균 제거 없이 root-mean-square로 마지막 feature 축을 정규화합니다."
    if bare == "GroupNorm":
        return "channel을 그룹으로 나누어 sample별 평균과 분산으로 정규화합니다."
    if bare.startswith("Dropout"):
        return "학습 중 무작위 원소 또는 channel을 0으로 만들고 나머지를 역확률로 보정합니다."
    if bare == "StochasticDepth":
        return "학습 중 sample 또는 batch 단위 residual branch를 확률적으로 생략합니다."
    if bare in {"RNN", "GRU", "LSTM"}:
        return f"다층·양방향 설정을 지원하는 {bare} sequence 계층입니다."
    if bare in {"RNNCell", "GRUCell", "LSTMCell"}:
        return f"sequence 한 step의 hidden state를 계산하는 {bare}입니다."
    if item.kind == "method":
        return f"{item.owner} 객체의 {bare} 동작을 수행합니다."
    if item.kind == "property":
        return f"{item.owner} 객체의 {bare} 상태를 읽기 전용으로 반환합니다."
    if item.kind == "class":
        return f"{bare} 신경망 구성요소를 생성합니다."
    return f"GPU Tensor에 {bare} 연산을 적용합니다."


def _parameter_description(name: str, descriptions: dict[str, str]) -> str:
    if name in descriptions:
        return descriptions[name]
    readable = name.replace("_", " ")
    return f"이 API의 {readable} 설정입니다. 허용 타입과 기본값은 시그니처를 따릅니다."


def _annotation(parameter: inspect.Parameter) -> str:
    if parameter.annotation is inspect.Parameter.empty:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return "tuple[Any, ...]"
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return "dict[str, Any]"
        return "Any"
    return inspect.formatannotation(parameter.annotation)


def _default(parameter: inspect.Parameter) -> str:
    if parameter.default is inspect.Parameter.empty:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            return "가변 위치 인자"
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return "가변 키워드 인자"
        return "필수"
    return repr(parameter.default).replace("<class 'cupy.float32'>", "float32")


def _return_text(item: ApiItem) -> str:
    bare = item.name.split(".")[-1]
    if item.kind == "class":
        return (
            f"설정된 {bare} 인스턴스. Parameter와 buffer는 지정 CUDA 장치에 생성됩니다."
        )
    if item.kind == "property":
        return f"현재 {bare} 값. 속성을 읽는 과정에서 GPU 수치 데이터를 CPU로 복사하지 않습니다."
    if bare in {"backward", "manual_seed", "save", "zero_grad", "step"}:
        return "None. 객체의 gradient, 난수 상태, 파일 또는 Parameter를 변경합니다."
    if bare in {"split", "chunk"}:
        return "원본과 같은 device에 있는 Tensor tuple. 분할 view의 역전파는 원래 위치로 gradient를 결합합니다."
    if bare == "topk":
        return "(values, indices) Tensor tuple. indices는 정수 dtype이며 자동미분 대상이 아닙니다."
    if bare in {"load", "state_dict"}:
        return "이름과 GPU Tensor를 연결한 순서 보존 mapping입니다."
    if bare in {"is_available", "is_grad_enabled"}:
        return "조건을 나타내는 Python bool입니다."
    if bare == "device_count":
        return "접근 가능한 장치 수를 나타내는 Python int입니다."
    if bare in {"parameters", "buffers", "children", "modules"}:
        return "등록 순서를 보존하는 iterator입니다."
    if bare.startswith("named_"):
        return "점으로 구분된 이름과 객체의 tuple을 순서대로 생성하는 iterator입니다."
    if bare in {"train", "eval", "to", "requires_grad_", "merge", "unmerge"}:
        return "호출 대상 객체 자신을 반환해 method chaining을 지원합니다."
    return "연산 결과 Tensor. 별도 설명이 없으면 입력과 같은 CUDA 장치에 있고 autograd 연결을 유지합니다."


def _behavior(item: ApiItem) -> list[str]:
    bare = item.name.split(".")[-1]
    notes = []
    if bare.startswith(("Conv", "MaxPool", "AvgPool", "Adaptive")):
        notes.append("입력은 batch와 channel 축 뒤에 1~3개의 공간축을 갖습니다.")
    elif bare in {"matmul", "mm", "bmm", "dot", "outer", "Linear", "Bilinear"}:
        notes.append(
            "내부 곱 차원이 맞지 않으면 두 입력 shape를 포함한 ValueError가 발생합니다."
        )
    elif "Loss" in bare or bare in LOSSES:
        notes.append(
            "reduction='none'은 축소 전 손실을, 'mean'과 'sum'은 scalar 손실을 반환합니다."
        )
    elif bare in ACTIVATIONS or CLASS_TO_FUNCTION.get(bare) in ACTIVATIONS:
        notes.append(
            "float16, float32, float64 입력을 지원하고 결과 dtype과 shape를 유지합니다."
        )
    elif item.kind in {"function", "method"}:
        notes.append(
            "broadcasting과 dtype 승격은 지원 범위 안에서 CuPy 규칙을 따릅니다."
        )
    if item.qualified.startswith("mytorch.optim") or bare in OPTIMIZERS:
        notes.append(
            "optimizer state와 모든 update는 Parameter가 있는 CUDA 장치에서 계산됩니다."
        )
    elif bare not in {"argmax", "argmin", "topk", "round", "sign"}:
        notes.append(
            "실수 입력이 requires_grad=True이고 gradient 모드가 켜져 있으면 backward 그래프를 기록합니다."
        )
    else:
        notes.append("정수 index 또는 비미분 결과는 requires_grad=False입니다.")
    notes.append(
        "잘못된 dtype, device, shape, 축 또는 범위 값은 자동 보정하지 않고 명확한 예외로 거부합니다."
    )
    return notes


def _class_members(value: type[Any], qualified: str) -> list[ApiItem]:
    result = []
    for name, member in value.__dict__.items():
        if name.startswith("_") or name in {"forward", "function"}:
            continue
        kind = None
        target = member
        if isinstance(member, property):
            kind = "property"
            target = member.fget
        elif isinstance(member, classmethod):
            kind = "method"
            target = member.__func__
        elif isinstance(member, staticmethod):
            kind = "method"
            target = member.__func__
        elif callable(member):
            kind = "method"
        if kind is not None and target is not None:
            result.append(
                ApiItem(
                    name=f"{value.__name__}.{name}",
                    qualified=f"{qualified}.{name}",
                    value=target,
                    kind=kind,
                    owner=value.__name__,
                )
            )
    return result


def _collect(module_name: str, include: list[str] | None = None) -> list[ApiItem]:
    module = importlib.import_module(module_name)
    items = []
    exported = set(getattr(module, "__all__", ()))
    members = (
        [(name, getattr(module, name)) for name in include]
        if include is not None
        else inspect.getmembers(module)
    )
    for name, value in members:
        if name.startswith("_"):
            continue
        defined_here = (
            getattr(value, "__module__", None) == module_name or name in exported
        )
        if inspect.isclass(value) and (defined_here or include is not None):
            kind = "class"
        elif inspect.isfunction(value) and (defined_here or include is not None):
            kind = "function"
        else:
            continue
        if module_name == "mytorch.tensor":
            qualified = f"mytorch.{name}"
        elif module_name in {"mytorch._autograd", "mytorch.serialization"}:
            qualified = f"mytorch.{name}"
        elif module_name.startswith("mytorch.nn.") and module_name not in {
            "mytorch.nn.functional",
            "mytorch.nn.init",
        }:
            qualified = f"mytorch.nn.{name}"
        elif module_name.startswith("mytorch.optim."):
            qualified = f"mytorch.optim.{name}"
        else:
            qualified = f"{module_name}.{name}"
        item = ApiItem(name=name, qualified=qualified, value=value, kind=kind)
        items.append(item)
        if kind == "class":
            items.extend(_class_members(value, qualified))
    return items


def _render_parameters(item: ApiItem, descriptions: dict[str, str]) -> str:
    parameters = _parameters(item)
    if not parameters:
        return '<p class="empty-note">별도 인자가 없습니다.</p>'
    rows = []
    for parameter in parameters:
        rows.append(
            "<tr>"
            f"<td><code>{_escape(parameter.name)}</code></td>"
            f"<td><code>{_escape(_annotation(parameter))}</code></td>"
            f"<td><code>{_escape(_default(parameter))}</code></td>"
            f"<td>{_escape(_parameter_description(parameter.name, descriptions))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table><thead><tr>'
        "<th>인자</th><th>타입</th><th>기본값</th><th>설명</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _render_api(item: ApiItem, descriptions: dict[str, str]) -> str:
    identifier = f"api-{_slug(item.qualified)}"
    signature = _signature(item)
    behavior = "".join(f"<li>{_escape(note)}</li>" for note in _behavior(item))
    badge = {
        "class": "Class",
        "function": "Function",
        "method": "Method",
        "property": "Property",
    }[item.kind]
    return f"""
      <article class="api-card searchable" id="{identifier}" data-search="{_escape(item.qualified)} {_escape(_summary(item))}">
        <header class="api-heading">
          <div>
            <span class="kind kind-{item.kind}">{badge}</span>
            <h2><code>{_escape(item.qualified)}</code></h2>
          </div>
          <button class="copy-link" type="button" data-copy="{identifier}" aria-label="이 API 링크 복사">링크 복사</button>
        </header>
        <p class="lead">{_escape(_summary(item))}</p>
        <section class="signature-block" aria-label="시그니처">
          <div class="section-label">Signature</div>
          <pre><code>{_escape(item.qualified)}{_escape(signature)}</code></pre>
        </section>
        <section>
          <h3>파라미터</h3>
          {_render_parameters(item, descriptions)}
        </section>
        <section class="api-grid">
          <div>
            <h3>반환값</h3>
            <p>{_escape(_return_text(item))}</p>
          </div>
          <div>
            <h3>동작과 주의사항</h3>
            <ul>{behavior}</ul>
          </div>
        </section>
      </article>
    """


def _nav(pages: list[dict[str, Any]], current: str, from_index: bool) -> str:
    links = []
    prefix = "api/" if from_index else ""
    groups: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        groups.setdefault(page["eyebrow"], []).append(page)
    for group, group_pages in groups.items():
        links.append(f'<div class="nav-group"><span>{_escape(group)}</span>')
        for page in group_pages:
            active = ' aria-current="page"' if page["slug"] == current else ""
            links.append(
                f'<a href="{prefix}{page["slug"]}.html"{active}>'
                f"{_escape(page['title'])}</a>"
            )
        links.append("</div>")
    return "".join(links)


def _shell(
    *,
    title: str,
    description: str,
    body: str,
    nav: str,
    asset_prefix: str,
    home_href: str,
    version: str,
) -> str:
    return f"""<!doctype html>
<html lang="ko" data-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{_escape(description)}">
  <meta name="generator" content="MyTorch docs builder">
  <title>{_escape(title)} · MyTorch {version}</title>
  <link rel="icon" href="{asset_prefix}assets/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="{asset_prefix}assets/css/docs.css">
</head>
<body>
  <a class="skip-link" href="#main">본문으로 건너뛰기</a>
  <header class="topbar">
    <button class="menu-button" type="button" aria-label="문서 메뉴 열기" aria-expanded="false">☰</button>
    <a class="brand" href="{home_href}"><span class="brand-mark">M</span><span>MyTorch</span><span class="version">v{version}</span></a>
    <div class="search-box">
      <label class="sr-only" for="site-search">API 검색</label>
      <input id="site-search" type="search" placeholder="API 검색…  /" autocomplete="off">
      <div id="search-results" class="search-results" hidden></div>
    </div>
    <button class="theme-button" type="button" aria-label="색상 테마 전환">◐</button>
  </header>
  <div class="site-layout">
    <aside class="sidebar" aria-label="문서 모듈 탐색">
      <a class="overview-link" href="{home_href}">문서 홈</a>
      {nav}
    </aside>
    <main id="main" class="content">{body}</main>
    <button class="back-to-top" type="button" aria-label="맨 위로">↑</button>
  </div>
  <footer>MyTorch {version} · GPU-only educational framework · API는 현재 소스 시그니처에서 생성되었습니다.</footer>
  <script src="{asset_prefix}assets/js/search-index.js"></script>
  <script src="{asset_prefix}assets/js/docs.js"></script>
</body>
</html>
"""


def _render_page(
    page: dict[str, Any],
    pages: list[dict[str, Any]],
    descriptions: dict[str, str],
    version: str,
) -> tuple[str, list[dict[str, str]]]:
    items = _collect(page["module"], page.get("include"))
    toc = "".join(
        f'<a href="#api-{_slug(item.qualified)}"><code>{_escape(item.name)}</code></a>'
        for item in items
    )
    concepts = "".join(f"<li>{_escape(value)}</li>" for value in page["concepts"])
    cards = "".join(_render_api(item, descriptions) for item in items)
    body = f"""
      <header class="page-header">
        <div class="eyebrow">{_escape(page["eyebrow"])} · {_escape(page["module"])}</div>
        <h1>{_escape(page["title"])}</h1>
        <p>{_escape(page["summary"])}</p>
        <div class="page-meta"><span>MyTorch {version}</span><span>{len(items)}개 API 항목</span><span>CUDA 전용</span></div>
      </header>
      <section class="concept-panel">
        <div><h2>이 모듈을 사용할 때 알아둘 점</h2><ul>{concepts}</ul></div>
        <div><h2>기본 예제</h2><pre><code>{_escape(page["example"])}</code></pre></div>
      </section>
      <nav class="page-toc" aria-label="이 페이지의 API"><strong>페이지 목차</strong><div>{toc}</div></nav>
      <div class="api-list">{cards}</div>
      <nav class="page-end"><a href="../index.html">← 전체 모듈 목록으로</a></nav>
    """
    search = [
        {
            "title": item.qualified,
            "url": f"api/{page['slug']}.html#api-{_slug(item.qualified)}",
            "module": page["module"],
            "summary": _summary(item),
            "kind": item.kind,
        }
        for item in items
    ]
    return body, search


def _render_index(pages: list[dict[str, Any]], version: str) -> str:
    cards = []
    for page in pages:
        cards.append(
            f"""
        <a class="module-card searchable" href="api/{page["slug"]}.html" data-search="{_escape(page["module"])} {_escape(page["title"])}">
          <span class="eyebrow">{_escape(page["eyebrow"])}</span>
          <h2>{_escape(page["title"])}</h2>
          <code>{_escape(page["module"])}</code>
          <p>{_escape(page["summary"])}</p>
          <span class="card-link">모듈 문서 보기 →</span>
        </a>
            """
        )
    return f"""
      <header class="home-header">
        <div class="eyebrow">한국어 API Reference</div>
        <h1>GPU에서 배우고,<br><span>GPU에서 끝나는 딥러닝.</span></h1>
        <p>MyTorch {version}의 Tensor, 자동미분, 신경망 계층, Transformer, BitNet, MoE와 optimizer를 실제 공개 시그니처에 맞춰 설명합니다.</p>
        <div class="home-actions"><a class="primary" href="api/tensor.html">Tensor부터 시작</a><a href="#modules">모듈 찾아보기</a></div>
      </header>
      <section class="quickstart">
        <div>
          <span class="eyebrow">Quick start</span>
          <h2>가장 작은 학습 루프</h2>
          <p>모델, 손실, optimizer가 모두 같은 CUDA 장치에서 연결됩니다. CPU fallback은 없습니다.</p>
        </div>
        <pre><code>import mytorch as mt

model = mt.nn.Sequential(
    mt.nn.Linear(2, 16), mt.nn.ReLU(), mt.nn.Linear(16, 1)
)
optimizer = mt.optim.AdamW(model.parameters(), lr=1e-3)

optimizer.zero_grad()
loss = mt.nn.functional.mse_loss(model(x), target)
loss.backward()
optimizer.step()</code></pre>
      </section>
      <section class="conventions">
        <h2>공통 규칙</h2>
        <div class="rule-grid">
          <article><strong>GPU only</strong><p>모든 forward, backward, optimizer state는 CUDA에 남습니다.</p></article>
          <article><strong>기본 dtype</strong><p>Python 실수와 sequence는 별도 지정이 없으면 float32입니다.</p></article>
          <article><strong>명시적 CPU 복사</strong><p>결과 확인은 Tensor.numpy() 또는 단일 원소의 item()을 사용합니다.</p></article>
          <article><strong>동적 autograd</strong><p>requires_grad와 현재 gradient mode에 따라 실행 시점에 그래프를 만듭니다.</p></article>
        </div>
      </section>
      <section id="modules" class="module-section">
        <div class="section-heading"><div><span class="eyebrow">Modules</span><h2>모듈별 상세 문서</h2></div><p>각 페이지는 독립적인 시그니처·인자·반환·오류 규칙을 제공합니다.</p></div>
        <div class="module-grid">{"".join(cards)}</div>
      </section>
      <section class="dtype-panel">
        <div><span class="eyebrow">Public constants</span><h2>지원 dtype</h2></div>
        <p><code>mt.float16</code> <code>mt.float32</code> <code>mt.float64</code> <code>mt.int8</code> <code>mt.int32</code> <code>mt.int64</code> <code>mt.uint8</code> <code>mt.bool</code></p>
      </section>
    """


def main() -> None:
    catalog = json.loads((CONTENT / "modules.json").read_text(encoding="utf-8"))
    descriptions = json.loads((CONTENT / "parameters.json").read_text(encoding="utf-8"))
    version = catalog["site"]["version"]
    if version != mt.__version__:
        raise RuntimeError(
            f"docs version {version} does not match package version {mt.__version__}"
        )
    pages = catalog["pages"]
    API_DIR.mkdir(parents=True, exist_ok=True)
    expected_api_pages = {f"{page['slug']}.html" for page in pages}
    for stale_page in API_DIR.glob("*.html"):
        if stale_page.name not in expected_api_pages:
            stale_page.unlink()
    search_index = []
    for page in pages:
        body, entries = _render_page(page, pages, descriptions, version)
        search_index.extend(entries)
        document = _shell(
            title=page["title"],
            description=page["summary"],
            body=body,
            nav=_nav(pages, page["slug"], from_index=False),
            asset_prefix="../",
            home_href="../index.html",
            version=version,
        )
        (API_DIR / f"{page['slug']}.html").write_text(document, encoding="utf-8")

    home = _shell(
        title="API Documentation",
        description=catalog["site"]["description"],
        body=_render_index(pages, version),
        nav=_nav(pages, "", from_index=True),
        asset_prefix="",
        home_href="index.html",
        version=version,
    )
    (DOCS_ROOT / "index.html").write_text(home, encoding="utf-8")
    search_json = json.dumps(search_index, ensure_ascii=False, separators=(",", ":"))
    (DOCS_ROOT / "assets" / "js" / "search-index.js").write_text(
        f"window.MYTORCH_SEARCH_INDEX={search_json};\n", encoding="utf-8"
    )
    print(f"Built {len(pages) + 1} pages with {len(search_index)} API entries")


if __name__ == "__main__":
    main()
