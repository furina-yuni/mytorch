# MyTorch

NVIDIA GPU에서만 수치 연산을 수행하는 교육용 Tensor 및 자동미분 프레임워크입니다.
현재 단계에서는 전용 conda 환경과 CUDA 동작 검증 기반을 제공합니다.

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
