# ----- 라이브러리 불러오기 -----
from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import jsona
import time
import threading
import numpy as np

# ----- 웹 서버 설정 -----
app = Flask(__name__)

# ----- 모델 설정 -----
# 모델 경로: 첫 번째 코드의 최신 모델 경로 사용
model_path = "C:/Users/user/Desktop/best_0613_2.pt"
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

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
HFOV = 46.05
FIRE_THRESHOLD_DEG = 0.5

# ----- 조준 관련 상수 -----
# 조준 보정값: 두 번째 코드의 튜닝된 값 사용
PITCH_FIRE_THRESHOLD_DEG = 0.1
PITCH_ADJUST_RANGE_FOR_WEIGHT = 30.0
AIMING_YAW_OFFSET_DEG = -0.5
PITCH_AIM_OFFSET_DEG = 1.2

# ----- 스캔 모드용 전역 변수 -----
SCAN_STEP_DEG = 45.0
PAUSE_SEC = 1.0
scan_origin_yaw = None
scan_index = 0
pause_start = None
scan_lap_count = 0
scan_done = False  # 버그 수정을 위해 스캔 완료 상태 변수 추가 및 초기화

# ----- 발사각(Pitch) 계산 모델 -----
# pitch ≈ c0*distance³ + c1*distance² + c2*distance + c3
PITCH_MODEL_COEFFS = [
    1.2e-05,
    -3.25e-03,
    3.914e-01,
    -11.6408
]
pitch_equation_model = np.poly1d(PITCH_MODEL_COEFFS)

def calculate_target_pitch(distance):
    """거리에 따라 필요한 포탄의 발사각(Pitch)을 계산합니다."""
    min_gun_pitch = -5.0
    max_gun_pitch = 9.75
    
    initial_calculated_pitch = pitch_equation_model(distance)
    final_target_pitch = np.clip(initial_calculated_pitch, min_gun_pitch, max_gun_pitch)
    
    if abs(final_target_pitch - initial_calculated_pitch) > 0.01:
        print(f"DEBUG (calculate_target_pitch): Distance: {distance:.2f}, Initial: {initial_calculated_pitch:.2f}, Clamped: {final_target_pitch:.2f}")
        
    return final_target_pitch

def _process_yolo_detection(image_file):
    """이미지를 받아 YOLO 탐지를 수행하고 결과를 반환합니다."""
    image_path = 'temp_image.jpg'
    try:
        image_file.save(image_path)
        results = model(image_path)
        yolo_detections = results[0].boxes.data.cpu().numpy()
        
        # 변수 이름 표준화: target_classes
        target_classes = {0: "tank", 1: "car"}
        filtered_results = []
        
        for box in yolo_detections:
            class_id = int(box[5])
            if class_id in target_classes:
                filtered_results.append({
                    'className': target_classes[class_id],
                    'bbox': [float(c) for c in box[:4]],
                    'confidence': float(box[4]),
                    'color': '#00FF00', 'filled': False, 'updateBoxWhileMoving': True
                })
        return filtered_results
    finally:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                print(f"Error: 임시 이미지 파일 {image_path} 삭제 실패: {e}")

def _get_filtered_lidar_points():
    """전역 변수 last_lidar_data에서 LiDAR 포인트를 필터링합니다."""
    if not last_lidar_data:
        return []
    raw_points = last_lidar_data.get('lidarPoints', [])
    if not isinstance(raw_points, list):
        return []
    return [p for p in raw_points if p.get('verticalAngle') == 0.0 and p.get('isDetected')]

def _find_distance_for_detection(detection, lidar_points, state):
    """탐지된 객체에 대해 가장 일치하는 LiDAR 거리 값을 찾습니다."""
    x1, _, x2, _ = detection['bbox']
    u_center = (x1 + x2) / 2.0
    phi_offset = (u_center / IMAGE_W - 0.5) * HFOV
    
    # 변수 이름 표준화: state['turret_yaw']
    phi_global_enemy = (state['turret_yaw'] + phi_offset + 360) % 360
    detection['phi'] = phi_global_enemy
    
    best_match = None
    smallest_angular_diff = float('inf')

    for point in lidar_points:
        # 변수 이름 표준화: state['turret_yaw']
        lidar_global_angle = (state['turret_yaw'] + point.get('angle', 0.0) + 360) % 360
        angular_diff = (lidar_global_angle - phi_global_enemy + 180 + 360) % 360 - 180
        
        if abs(angular_diff) < smallest_angular_diff:
            smallest_angular_diff = abs(angular_diff)
            best_match = point
            
    # 버그 수정: for 루프가 끝난 후, 최종적으로 찾은 best_match의 거리를 반환
    if best_match:
        # 변수 이름 표준화: 'distance'
        return best_match.get('distance')
        
    return None

def _log_data(filepath, data):
    """주어진 데이터를 JSON 형태로 로그 파일에 기록합니다."""
    with log_lock, open(filepath, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

@app.route('/detect', methods=['POST'])
def detect():
    """메인 탐지 로직: 이미지와 LiDAR 데이터를 융합합니다."""
    global last_lidar_data, last_enemy_data
    
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({"error": "No image received"}), 400

    current_state = {'time': time.time(), 'turret_yaw': 0.0, 'body_yaw': 0.0}
    if last_lidar_data:
        current_state['time'] = last_lidar_data.get('time', current_state['time'])
        # 변수 이름 표준화: turret_yaw
        current_state['turret_yaw'] = last_lidar_data.get('playerTurretX', 0.0)
        current_state['body_yaw'] = last_lidar_data.get('playerBodyX', 0.0)

    yolo_detections = _process_yolo_detection(image_file)
    _log_data('logs/detections.json', {'timestamp': current_state['time'], 'turretYaw': current_state['turret_yaw'], 'detections': yolo_detections})

    lidar_points = _get_filtered_lidar_points()

    # 변수 이름 표준화: enemy_distances, distance
    enemy_distances = []
    for det in yolo_detections:
        if det['className'] == 'tank':
            distance = _find_distance_for_detection(det, lidar_points, current_state)
            if distance is None:
                distance = 50.0  # 기본 거리값
            enemy_distances.append({
                'phi': det['phi'],
                'distance': distance,
                'body_size': ENEMY_BODY_SIZE,
                'turret_size': ENEMY_TURRET_SIZE
            })

    last_enemy_data = {'timestamp': current_state['time'], 'enemies': enemy_distances}
    _log_data('logs/enemy.json', last_enemy_data)

    return jsonify(yolo_detections)

@app.route('/info', methods=['POST'])
def info():
    """시뮬레이터로부터 탱크 상태 정보를 주기적으로 수신합니다."""
    global last_lidar_data
    data = request.get_json(force=True) or {}
    last_lidar_data = data
    return jsonify({'status': 'success', 'control': ''})

@app.route('/get_action', methods=['POST'])
def get_action():
    """탐지된 정보를 바탕으로 탱크의 행동을 결정합니다."""
    global last_lidar_data, last_enemy_data
    global scan_origin_yaw, scan_index, pause_start, scan_lap_count, scan_done

    # 변수 이름 표준화: turret_yaw, turret_pitch
    turret_yaw = last_lidar_data.get('playerTurretX', 0.0)
    turret_pitch = last_lidar_data.get('playerTurretY', 0.0)

    cmd = {'moveWS':{'command':'STOP','weight':1.0}, 'moveAD':{'command':'','weight':0.0},
            'turretQE':{'command':'','weight':0.0}, 'turretRF':{'command':'','weight':0.0}, 'fire':False}

    # 스캔이 완료되었으면 아무것도 하지 않음
    if scan_done:
        return jsonify(cmd)

    # 적이 있으면 조준/발사
    if last_enemy_data and last_enemy_data.get('enemies'):
        # 변수 이름 표준화: target, dist, pitch_tgt
        target = min(last_enemy_data['enemies'], key=lambda e: e['distance'])
        phi_tgt = target['phi']
        dist = target['distance']
        pitch_tgt = calculate_target_pitch(dist) + PITCH_AIM_OFFSET_DEG

        delta_yaw = ((phi_tgt - turret_yaw + 180) % 360) - 180
        yaw_weight = min(abs(delta_yaw) / 180.0, 1.0) * 5
        delta_pitch = pitch_tgt - turret_pitch
        pitch_weight = min(abs(delta_pitch) / PITCH_ADJUST_RANGE_FOR_WEIGHT, 1.0) * 5

        if abs(delta_yaw) <= FIRE_THRESHOLD_DEG and abs(delta_pitch) <= PITCH_FIRE_THRESHOLD_DEG:
            cmd['fire'] = True
        else:
            if abs(delta_yaw) > FIRE_THRESHOLD_DEG:
                cmd['turretQE'] = {'command': 'E' if delta_yaw > 0 else 'Q', 'weight': yaw_weight}
            if abs(delta_pitch) > PITCH_FIRE_THRESHOLD_DEG:
                cmd['turretRF'] = {'command': 'R' if delta_pitch > 0 else 'F', 'weight': pitch_weight}
    
    # 적이 없으면 스캔 모드 (빠져있던 로직 추가)
    else:
        # 최초 스캔 진입 시, 현재 각도를 시작점으로 설정
        if scan_origin_yaw is None:
            scan_origin_yaw = -45.0 # 시작각도 -90도로 처음시작 : 정면타격후 왼쪽으로 돌아가는거부터 시작
            scan_index = 0
            pause_start = None
            scan_lap_count = 0

        # 목표 각도 계산
        
        target_yaw = (scan_origin_yaw + SCAN_STEP_DEG * scan_index) % 360
        delta = ((target_yaw - turret_yaw + 180) % 360) - 180

        # 목표 전까지 회전
        if abs(delta)>1.0:
            pause_start = None
            speed_weight = min(abs(delta)/30.0,0.5)
            cmd['turretQE'] = {'command':'E' if delta>0 else 'Q','weight':speed_weight}
        else:
            # 목표 도달 후 대기
            if pause_start is None:
                pause_start = time.time()
            if time.time() - pause_start < PAUSE_SEC:
                cmd['turretQE'] = {'command':'','weight':0.0}
            else:
                scan_index += 1
                if scan_index >= int(360/SCAN_STEP_DEG):
                    scan_index     = 0
                    scan_lap_count += 1
                    

                # 2바퀴 돌았으면 정면 복귀 로직 -> # 1바퀴 돌고 종료
                if scan_lap_count >= 1:
                    delta_to_origin = ((scan_origin_yaw - turret_yaw + 180) % 405) - 180
                    if abs(delta_to_origin)>1.0:
                        speed_weight = min(abs(delta_to_origin)/30.0,0.5)
                        cmd['turretQE'] = {'command':'E' if delta_to_origin>0 else 'Q','weight':speed_weight}
                    else:
                        scan_done = True
                        cmd['turretQE'] = {'command':'','weight':0.0}
                    return jsonify(cmd)
                pause_start = None

    return jsonify(cmd)


@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    return jsonify({'status': 'OK', 'message': 'Bullet impact data received'})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    return jsonify({'status': 'OK', 'destination': {}})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST'])
def collision():
    return jsonify({'status': 'success', 'message': 'Collision data received'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        'startMode': 'start', 
        'blStartX': 60, 'blStartY': 10, 'blStartZ': 27.23,
        'rdStartX': 59, 'rdStartY': 10, 'rdStartZ': 280, 
        'trackingMode': False,'detactMode': True, 'logMode': True, 
        'enemyTracking': False, 'saveSnapshot': False,'saveLog': True,
        'saveLidarData': True, 'lux': 30000,
        'player_body_size': PLAYER_BODY_SIZE, 
        'player_turret_size': PLAYER_TURRET_SIZE,
        'enemy_body_size': ENEMY_BODY_SIZE, 
        'enemy_turret_size': ENEMY_TURRET_SIZE
    }
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    return jsonify({'control': ''})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
