# 🧭 자율주행 및 객체 탐지 기반 Flask API 서버

이 프로젝트는 **WSL2 기반 Ubuntu 환경**에서 YOLOv8 객체 탐지와 A* 알고리즘을 활용한 자율주행 기능을 구현한 Flask 기반 API 서버입니다. 
Jupyter Lab 환경에서 실험 및 개발을 진행하며, 실시간으로 장애물을 업데이트하고 목표 지점까지의 경로를 자동 탐색하여 이동 제어 명령을 생성합니다.

---

## 📁 프로젝트 구성

- **YOLOv8 객체 탐지 모듈**: 이미지 업로드를 통해 객체 탐지 수행 (`/detect`)
- **A* 기반 경로 탐색 모듈**: 장애물 회피 및 목적지까지 경로 탐색 (`/get_action`)
- **Flask 서버 기반 API**: 시뮬레이터와의 연동 지원

---

## ⚙️ 개발 환경 가이드 (WSL2 + Ubuntu + Anaconda + Jupyter Lab + YOLOv8 + GPU 지원)

### ✅ 환경 요약

| 항목       | 구성 내용                                      |
|------------|-----------------------------------------------|
| OS         | Windows 10/11 + WSL2 + Ubuntu 20.04           |
| GPU        | NVIDIA GPU (WSL2 CUDA 드라이버 필수)          |
| 가상환경   | Anaconda (Python 3.10)                        |
| 프레임워크 | YOLOv8 (Ultralytics) + PyTorch + CUDA         |
| 인터페이스 | Jupyter Lab + Flask API 연동 가능             |

---

### 1️⃣ WSL2 설치 및 GPU 설정

```powershell
wsl --install
```

- Ubuntu 20.04 설치 (Microsoft Store 사용)
- [NVIDIA 공식 드라이버 설치](https://www.nvidia.com/Download/index.aspx)
- 설치 확인:

```bash
wsl
nvidia-smi
```

---

### 2️⃣ Ubuntu 내 필수 패키지 설치

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install wget curl git unzip -y
```

---

### 3️⃣ Anaconda 설치

```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
bash Anaconda3-2023.07-1-Linux-x86_64.sh
source ~/.bashrc
```

---

### 4️⃣ Conda 가상환경 및 Jupyter Lab 설치

```bash
conda create -n yolov8_gpu python=3.10 -y
conda activate yolov8_gpu

conda install -c conda-forge jupyterlab -y
```

---

### 5️⃣ PyTorch (CUDA 지원) 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

### 6️⃣ YOLOv8 설치

```bash
pip install ultralytics
```

모델 테스트:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('image.jpg')
```

---

### 7️⃣ Jupyter Lab 실행

```bash
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

브라우저 접속:
```
http://localhost:8888/?token=...
```

---

### 8️⃣ GPU 확인

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # NVIDIA GPU 이름 출력
```

---

# 📌 Git 작업 시 유의사항

## 🔧 시뮬레이터 버전 안내
- 현재 사용 중인 Tank Challenge 시뮬레이터 버전: **2.2.3**
- 해당 버전은 **최신 안정성 문제로 인해 구버전 유지 중**

---

## ✅ 커밋 작성 규칙
- 커밋 시 반드시 **변동 사항에 대한 상세한 설명**을 작성할 것
- **불명확하거나 일반적인 커밋 메시지 금지** (예: `fix`, `update` 등 단독 사용 지양)
- 커밋 내용은 **주기적으로 확인**하여 팀원 간 진행 상황을 공유할 것

---

## ⚠️ 컨플릭트(Conflict) 발생 시
- 충돌 발생 시 **작업자 간 사전 협의**를 통해 코드 통일
- **임의 수정 또는 무단 병합 절대 금지**
- 협의를 통해 최종 코드 방향을 결정 후 병합 수행

---

## 🛠️ 베이스 코드 변경 시
- 기존 베이스 코드를 **삭제/재작성해야 하는 경우**, 반드시 **해당 기능 담당자 간 상의 후 결정**
- 사전 논의 없이 전체 코드 변경 금지

---

## 🗂️ 브랜치 작업 분할
- 현재 작업 브랜치는 다음과 같이 기능별로 분할되어 있음:
  - `autonomous-driving`: 자율주행 기능 개발
  - `autonomous-exploration`: 자율탐색 기능 개발
- 각 브랜치의 목적에 맞게 코드를 구성하고, **기능 간 분리를 유지**할 것

---

> 🧭 협업 시 신뢰와 커뮤니케이션을 최우선으로 생각해주세요.


---
## 시뮬레이터 Lidar 참고

![image](https://github.com/user-attachments/assets/9b9ea699-5ae1-4489-8a56-f738a5b8af8f)


## 📮 문의
본 프로젝트는 군집 자율 탐색 및 교전 로직 구현을 위한 내부 R&D 용도로 제작되었습니다.
추가 정보나 협업 요청은 관리자에게 별도 문의 바랍니다.
