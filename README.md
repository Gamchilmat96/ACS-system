# ACS-system
🛡️ 전차 시뮬레이터 기반 자율주행 시스템
본 프로젝트는 Flask 서버와 A* 알고리즘을 기반으로 한 전차 자율주행 시뮬레이터입니다.
시뮬레이터는 전장 환경을 반영한 격자 맵을 활용하여 장애물을 회피하고, 목표 지점까지 자율적으로 주행하는 기능을 구현합니다.

추후 YOLOv8 객체 탐지 모델과 결합하여 적 전차 탐지 및 자율 교전 기능까지 확장할 예정이며, 현재는 주행 알고리즘과 서버 API 안정화에 초점을 맞추고 있습니다.

개발 환경은 Jupyter Lab 기반이며, Flask 서버를 통해 시뮬레이터와 실시간으로 연동됩니다.
GitHub를 통한 협업이 이루어지고 있으며, 모든 작업은 브랜치 기반으로 이루어져야 합니다.


⚙️ 개발 환경 구성 가이드 (Windows 10 + WSL2 + Ubuntu + YOLOv8)
이 프로젝트는 Windows 10 환경에서 WSL2를 활용한 Ubuntu 가상환경에서 개발되며,
Jupyter Lab 상에서 YOLOv8 객체 탐지 모델 학습 및 Flask 서버 연동을 통해 자율 주행 테스트를 진행합니다.

1️⃣ WSL2 + Ubuntu 설치

```bash
# WSL2 설치 (관리자 PowerShell)
wsl --install

# 재부팅 후 Ubuntu 설치
wsl --install -d Ubuntu-20.04
```
이후 Ubuntu 터미널을 실행하여 초기 사용자 및 비밀번호 설정

2️⃣ Ubuntu에서 초기 패키지 업데이트

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install wget curl git unzip -y

```

3️⃣ Anaconda 설치 (Ubuntu 내부)
```bash
# Anaconda 다운로드 및 설치
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
bash Anaconda3-2023.07-1-Linux-x86_64.sh

# 설치 완료 후
source ~/.bashrc
```
설치 중 경로는 기본 추천값(/home/사용자/anaconda3)을 사용하세요.

4️⃣ 가상환경 생성 및 Jupyter Lab 설치
```bash
# YOLOv8용 가상환경 생성
conda create -n {가상환경 이름} python=3.10 -y
conda activate {가상환경 이름}

# Jupyter Lab 설치
conda install -c conda-forge jupyterlab -y
```

5️⃣ YOLOv8 설치 (Ultralytics)
```bash
pip install ultralytics
```

6️⃣ Jupyter Lab 실행
```bash
jupyter lab --no_browser --ip=0.0.0.0
```

| 항목     | 내용                               |
| ------ | -------------------------------- |
| OS     | Windows 10 + WSL2 + Ubuntu 20.04 |
| Python | 3.10 (Anaconda 가상환경)             |
| 프레임워크  | Jupyter Lab + Ultralytics YOLOv8 |
| 사용 모델  | YOLOv8 (`best.pt` 사용 가능)         |
| 서버     | Flask 기반 서버 연동                   |
