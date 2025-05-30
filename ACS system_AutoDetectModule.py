# ----- 라이브러리 불러오기 -----
from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import json
import time
import threading
import math

# ----- 웹 서버 설정 -----
app = Flask(__name__)

# ----- 모델 설정 -----
model_path = "best.pt"
model = YOLO(model_path)

# ----- 탱크 크기 정보 -----
PLAYER_BODY_SIZE   = (3.667, 1.582, 8.066)
PLAYER_TURRET_SIZE = (3.297, 2.779, 5.891)
ENEMY_BODY_SIZE    = (3.303, 1.131, 6.339)
ENEMY_TURRET_SIZE  = (2.681, 3.094, 2.822)

# ----- 전역 변수 및 동기화 설정 -----
last_lidar_data = None
last_enemy_data = None
log_lock = threading.Lock()

# ----- 설정값 -----
IMAGE_W = 2560
HFOV    = 90.0
FIRE_THRESHOLD_DEG = 5.0

@app.route('/detect', methods=['POST'])
def detect():
    """
    1) 시뮬레이터에서 전송된 스크린샷(image) 받아 저장
    2) YOLO 모델로 car/tank 탐지
    3) detections.json에 탐지된 모든 객체 정보(타임스탬프, 포탑 각도 포함) 기록
    4) 탐지된 tank 객체에 대해 LiDAR 데이터와 매칭하여 enemy.json에 거리 및 크기 정보 기록
    5) 탐지된 객체들의 바운딩 박스 정보를 클라이언트(시뮬레이터)로 반환
    """
    global last_lidar_data, last_enemy_data

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: "car", 1: "tank"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(c) for c in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': True
            })

    ts = time.time()
    turret_yaw = 0.0
    if last_lidar_data:
        ts = last_lidar_data.get('time', ts)
        turret_yaw = last_lidar_data.get('playerTurretY', turret_yaw)

    detection_record = {'timestamp': ts, 'turretYaw': turret_yaw, 'detections': filtered_results}
    os.makedirs('logs', exist_ok=True)
    with log_lock, open('logs/detections.json', 'a', encoding='utf-8') as f:
        json.dump(detection_record, f, ensure_ascii=False)
        f.write('\n')

    enemy_distances = []
    if last_lidar_data:
        lidar_pts = last_lidar_data.get('lidarPoints', [])
        for det in filtered_results:
            if det['className'] == 'tank':
                x1, _, x2, _ = det['bbox']
                u_center = (x1 + x2) / 2.0
                u_norm = u_center / IMAGE_W
                phi_offset = (u_norm - 0.5) * HFOV
                phi_global = (turret_yaw + phi_offset) % 360

                if lidar_pts:
                    best = min(lidar_pts, key=lambda p: abs(p.get('angle',0) - phi_global))
                    dist = best.get('distance')
                    if dist is not None:
                        enemy_distances.append({
                            'phi': phi_global,
                            'distance': float(dist),
                            'body_size': ENEMY_BODY_SIZE,
                            'turret_size': ENEMY_TURRET_SIZE
                        })

    last_enemy_data = {'timestamp': ts, 'enemies': enemy_distances}
    with log_lock, open('logs/enemy.json', 'a', encoding='utf-8') as f:
        json.dump(last_enemy_data, f, ensure_ascii=False)
        f.write('\n')

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    """
    1) 시뮬레이터로부터 플레이어 상태 및 LiDAR 등 센서 데이터(JSON) 수신
    2) 수신된 데이터를 전역 변수 last_lidar_data에 저장
    3) 성공 상태 및 빈 control 명령 반환
    """
    global last_lidar_data
    data = request.get_json(force=True)
    last_lidar_data = data
    return jsonify({'status': 'success', 'control': ''})

@app.route('/get_action', methods=['POST'])
def get_action():
    """
    1) 시뮬레이터로부터 현재 플레이어의 위치(position) 및 포탑(turret) 상태 수신
    2) 가장 최근에 탐지된 적(last_enemy_data) 정보 기반으로 행동 결정
    3) 가장 가까운 적 탱크를 향해 포탑을 회전시키고, 조준 완료 시 발사 명령 생성
    4) 계산된 이동/회전/발사 명령을 JSON 형태로 시뮬레이터에 반환
    """
    global last_enemy_data
    data = request.get_json(force=True)
    position = data.get('position', {})
    turret   = data.get('turret', {})

    x0 = position.get('x', 0.0)
    z0 = position.get('z', 0.0)
    turret_yaw = turret.get('y', 0.0)

    cmd = {'moveWS': {'command': 'STOP', 'weight': 1.0},
           'moveAD': {'command': '', 'weight': 0.0},
           'turretQE': {'command': '', 'weight': 0.0},
           'turretRF': {'command': '', 'weight': 0.0},
           'fire': False}

    if last_enemy_data and last_enemy_data.get('enemies'):
        target = min(last_enemy_data['enemies'], key=lambda e: e['distance'])
        phi_target = target['phi']
        # dist = target['distance'] # 주석 처리: 현재 사용되지 않음

        delta = ((phi_target - turret_yaw + 180) % 360) - 180
        weight = min(abs(delta) / 180.0, 1.0)
        if delta > FIRE_THRESHOLD_DEG:
            cmd['turretQE'] = {'command': 'E', 'weight': weight}
        elif delta < -FIRE_THRESHOLD_DEG:
            cmd['turretQE'] = {'command': 'Q', 'weight': weight}
        else:
            cmd['fire'] = True

    return jsonify(cmd)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    """
    시뮬레이터에서 발사된 총알의 결과(예: 명중 여부)를 수신하기 위한 엔드포인트.
    현재는 단순히 수신 확인 메시지만 반환.
    """
    return jsonify({'status': 'OK', 'message': 'Bullet impact data received'})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    """
    플레이어 탱크의 목표 지점 설정을 위한 엔드포인트.
    현재는 단순히 수신 확인 및 빈 목적지 정보 반환.
    """
    return jsonify({'status': 'OK', 'destination': {}})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """
    시뮬레이터 내 장애물 정보 업데이트를 위한 엔드포인트.
    현재는 단순히 수신 확인 메시지만 반환.
    """
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST'])
def collision():
    """
    플레이어 탱크의 충돌 발생 정보 수신을 위한 엔드포인트.
    현재는 단순히 수신 확인 메시지만 반환.
    """
    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    """
    시뮬레이터 초기 실행 시 필요한 각종 설정값을 제공하는 엔드포인트.
    팀 시작 위치, 각종 모드 활성화 여부, 탱크 크기 정보 등을 JSON으로 반환.
    """
    config = {
        'startMode': 'start',
        'blStartX': 60, 'blStartY': 10, 'blStartZ': 27.23,
        'rdStartX': 59, 'rdStartY': 10, 'rdStartZ': 280,
        'trackingMode': True, 'detactMode': True, 'logMode': True,
        'enemyTracking': True, 'saveSnapshot': True, 'saveLog': True,
        'saveLidarData': True, 'lux': 30000,
        'player_body_size': PLAYER_BODY_SIZE,
        'player_turret_size': PLAYER_TURRET_SIZE,
        'enemy_body_size': ENEMY_BODY_SIZE,
        'enemy_turret_size': ENEMY_TURRET_SIZE
    }
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    """
    시뮬레이션 또는 게임 에피소드의 시작을 알리는 엔드포인트.
    현재는 빈 control 명령을 반환.
    """
    return jsonify({'control': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)