# 🧭 자율주행 및 객체 탐지 기반 Flask API 서버

이 프로젝트는 **WSL2 기반 Ubuntu 환경**에서 YOLOv8 객체 탐지와 A* 알고리즘을 활용한 자율주행 기능을 구현한 Flask 기반 API 서버입니다. 
Jupyter Lab 환경에서 실험 및 개발을 진행하며, 실시간으로 장애물을 업데이트하고 목표 지점까지의 경로를 자동 탐색하여 이동 제어 명령을 생성합니다.

---

## 📁 프로젝트 구성

- **YOLOv8 객체 탐지 모듈**: 이미지 업로드를 통해 객체 탐지 수행 (`/Auto_Detect`)
- **A* 기반 경로 탐색 모듈**: 장애물 회피 및 목적지까지 경로 탐색 (`/Auto_Driving`)
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

=======
### 이미지 데이터

YOLOv8 학습을 위한 이밎, 라벨 데이터는 해당 구글 드라이브 링크에 포함되어 있는 "data_set"를 사용하면 됩니다.
https://drive.google.com/drive/folders/1EgZXMk7Odpa1UGMKK0g0PtYBIBWwclvr?usp=drive_link

### YOLOv8 학습모델

YOLOv8 학습 모델은 해당 구글 드라이브 링크에 포함되어 있는 "best.pt"를 사용하면 됩니다.
https://drive.google.com/file/d/1r6eOJzkgqLAUa0SOGsp6F-SsBC6zXglW/view?usp=drive_link

## 시뮬레이터 설정

현재 Tank Challenge 시뮬레이터는 시뮬레이터 설정으로 Lidar sensor의 탐지 범위를 설정할 수 있습니다.
![image](https://github.com/user-attachments/assets/62da08a5-6fe1-4201-9b50-372b43cc0ab1)

위 사진과 같이 Y position, Channel, Minimap Channel, Max Distance, Lidar position으로 나누어집니다.

- Y position : Lidar sensor의 위치를 Y축 기준으로 세팅할 수 있습니다. 해당 값이 지나치게 낮으면 지면에 크게 영향을 받고 반대로 지나치게 높으면 재대로 탐지를 못할 가능성이 커집니다.
- Channel : Lidar sensor는 기본으로 상하로 22.5도 범위를 탐색합니다. Channel 값을 통해 해당 범위를 얼마나 세분화 할 수 있는지 정할 수 있습니다. 값이 지나치게 낮으면 탐지 가능성이 작아지고 값이 지나치게 크면 오브젝트 탐지를 구분하지 못할 수 있습니다.
- Minimap Channel : 시뮬레이터 미니맵으로 표시할 채널을 설정할 수 있습니다. 시각화와 관련한 설정이기에 실제 성능에 영향을 주지 않습니다.
- Max Distance : Lidar sensor가 감지할 수 있는 최대거리를 설정하는 값입니다. 
- Lidar position : Lidar sensor의 기본 위치를 Turret으로 할지 Body에 할지 정하는 설정입니다. Turret을 선택하게 되면 라이다의 방향이 Turret의 회전에 따라서 변경되고 Body를 선택하면 Turret의 회전과 상관없이 전차 몸체의 방향에 따라 Foward가 설정 됩니다.

현재 추천하는 세팅은 Y position = 1, Channel = 11, Lidar Position =  Turret 입니다.

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
  - `ACS system_AutoDriveModule`: 자율주행 기능 개발
  - `ACS system_AutoDetectModule`: 자율탐색 기능 개발
- 각 모듈의 목적에 맞게 코드를 구성하고, **기능 간 분리를 유지**할 것

---

> 🧭 협업 시 신뢰와 커뮤니케이션을 최우선으로 생각해주세요.

---
## 📮 문의
본 프로젝트는 군집 자율 탐색 및 교전 로직 구현을 위한 내부 R&D 용도로 제작되었습니다.
추가 정보나 협업 요청은 관리자에게 별도 문의 바랍니다.
