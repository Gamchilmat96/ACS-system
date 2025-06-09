# ----- 라이브러리 불러오기 -----
from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import json
import time
import threading
import math
import glob # 파일 검색을 위해 추가
import re   # 파일명 파싱을 위해 추가
import numpy as np # 다항식 계산 및 clip 함수 사용을 위해 추가

# ----- 웹 서버 설정 -----
app = Flask(__name__)

# ----- 모델 설정 -----
model_path = "best.pt" # 사용자의 모델 경로
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()


# ----- 탱크 크기 정보 -----
PLAYER_BODY_SIZE    = (3.667, 1.582, 8.066)
PLAYER_TURRET_SIZE = (3.297, 2.779, 5.891)
ENEMY_BODY_SIZE     = (3.303, 1.131, 6.339)
ENEMY_TURRET_SIZE   = (2.681, 3.094, 2.822)

# ----- 전역 변수 및 동기화 설정 -----
last_lidar_data = None # /info 엔드포인트를 통해 업데이트됨
last_enemy_data = None
log_lock = threading.Lock()

# ----- 설정값 -----
IMAGE_W = 2560
HFOV    = 46.05
FIRE_THRESHOLD_DEG = 1.0 # 수평 발사 허용 오차 (도)

# ----- LiDAR 데이터 경로 (WSL 환경 기준) -----
LIDAR_DATA_DIR = "/mnt/c/Users/bok7z/OneDrive/문서/Tank Challenge/lidar_data" # 사용자가 제공한 경로

# ----- 발사각(Pitch) 계산 모델 계수 -----
# pitch ≈ c0*distance³ + c1*distance² + c2*distance + c3
PITCH_MODEL_COEFFS = [
    1.18614662e-05,  # c0 (distance³의 계수)
    -3.20931503e-03, # c1 (distance²의 계수)
    3.87703588e-01,  # c2 (distance의 계수)
    -11.55315302e+00  # c3 (상수항)
]
pitch_equation_model = np.poly1d(PITCH_MODEL_COEFFS)

# ----- 새로운 조준 관련 상수 -----
PITCH_FIRE_THRESHOLD_DEG = 0.5
PITCH_ADJUST_RANGE_FOR_WEIGHT = 30.0

def calculate_target_pitch(distance):
    min_gun_pitch = -5.0  # 예시: 실제 탱크의 최소 발사각 (필요시 실제값으로 수정)
    max_gun_pitch = 9.75  # 예시: 실제 탱크의 최대 발사각 (필요시 실제값으로 수정)
    initial_calculated_pitch = pitch_equation_model(distance)
    final_target_pitch = np.clip(initial_calculated_pitch, min_gun_pitch, max_gun_pitch)
    
    # 만약 clamping으로 인해 값이 변경되었다면 로그를 남겨 확인 (디버깅용)
    if abs(final_target_pitch - initial_calculated_pitch) > 0.01: # 아주 작은 차이는 무시
        print(f"DEBUG (calculate_target_pitch): Distance: {distance:.2f}, Initial Calc Pitch: {initial_calculated_pitch:.2f}, Clamped Target Pitch: {final_target_pitch:.2f} (Limits: {min_gun_pitch:.2f} to {max_gun_pitch:.2f})")
        
    return final_target_pitch

# ----- Helper function to get the latest LiDAR data file -----
def get_latest_lidar_data_filepath(directory):
    latest_file = None; latest_t_val = -1; latest_frame_val = -1
    pattern = re.compile(r"LidarData_t(\d+)_(\d+)\.json")
    try:
        if not os.path.isdir(directory):
            print(f"Warning: LiDAR 데이터 디렉토리를 찾을 수 없거나 디렉토리가 아닙니다: {directory}"); return None
        filenames = os.listdir(directory)
        if not filenames:
            print(f"Warning: LiDAR 데이터 디렉토리에 파일이 없습니다: {directory}"); return None
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                t_val, frame_val = int(match.group(1)), int(match.group(2))
                if t_val > latest_t_val or (t_val == latest_t_val and frame_val > latest_frame_val):
                    latest_t_val, latest_frame_val, latest_file = t_val, frame_val, os.path.join(directory, filename)
        if not latest_file: print(f"Warning: {directory} 에서 패턴에 맞는 LiDAR 파일을 찾지 못했습니다.")
    except Exception as e: print(f"LiDAR 디렉토리 접근 중 오류 발생 {directory}: {e}"); return None
    return latest_file

@app.route('/detect', methods=['POST'])
def detect():
    global last_lidar_data, last_enemy_data
    image_file = request.files.get('image')
    if not image_file: return jsonify({"error": "No image received"}), 400
    image_path = 'temp_image.jpg'; image_file.save(image_path)
    results = model(image_path); yolo_detections = results[0].boxes.data.cpu().numpy()
    target_classes = {0: "tank", 1: "car"} # 사용자 정의
    filtered_results = []
    for box in yolo_detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id], 'bbox': [float(c) for c in box[:4]], 
                'confidence': float(box[4]), 'color': '#00FF00', 'filled': False, 'updateBoxWhileMoving': True 
            })
    current_ts = time.time()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # 사용자 확인: X가 Yaw(좌우), Y가 Pitch(상하)
    # /detect 에서는 last_lidar_data의 playerTurretX를 turret_yaw로 사용
    turret_yaw_from_lidar = 0.0 
    player_body_yaw = 0.0 
    if last_lidar_data:
        current_ts = last_lidar_data.get('time', current_ts)
        turret_yaw_from_lidar = last_lidar_data.get('playerTurretX', 0.0) # X가 Yaw이므로 playerTurretX 사용
        player_body_yaw = last_lidar_data.get('playerBodyY', 0.0) # 차체 Yaw는 playerBodyY로 가정 (일반적)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    detection_record = {'timestamp': current_ts, 'turretYaw': turret_yaw_from_lidar, 'detections': filtered_results}
    os.makedirs('logs', exist_ok=True)
    with log_lock, open('logs/detections.json', 'a', encoding='utf-8') as f:
        json.dump(detection_record, f, ensure_ascii=False); f.write('\n')
    enemy_distances = []
    latest_lidar_filepath = get_latest_lidar_data_filepath(LIDAR_DATA_DIR)
    device_lidar_points_from_file = []
    if latest_lidar_filepath:
        try:
            with open(latest_lidar_filepath, 'r', encoding='utf-8') as f_lidar:
                loaded_json_data = json.load(f_lidar)
                if isinstance(loaded_json_data, dict) and 'data' in loaded_json_data and isinstance(loaded_json_data['data'], list):
                    device_lidar_points_from_file = loaded_json_data['data']
                elif isinstance(loaded_json_data, list): device_lidar_points_from_file = loaded_json_data
                else: print(f"Warning [/detect]: {latest_lidar_filepath} JSON 'data' 키 없음 또는 예상 구조 아님.")
        except Exception as e: print(f"Error [/detect]: LiDAR 파일 처리 오류: {e}")
    filtered_device_lidar_points = []
    if device_lidar_points_from_file: 
        for p in device_lidar_points_from_file:
            if p.get('verticalAngle') == 0.0 and p.get('isDetected') is True:
                filtered_device_lidar_points.append(p)
        if not filtered_device_lidar_points and device_lidar_points_from_file: print("Warning [/detect]: LiDAR 포인트 필터링 결과 없음.")
    elif latest_lidar_filepath: pass 
    else: print(f"Warning [/detect]: {LIDAR_DATA_DIR} 에서 최신 LiDAR 파일 못찾음.")
    if filtered_device_lidar_points and last_lidar_data:
        for det in filtered_results:
            if det['className'] == 'tank':
                x1, _, x2, _ = det['bbox']
                u_center = (x1 + x2) / 2.0; u_norm = u_center / IMAGE_W
                phi_offset = (u_norm - 0.5) * HFOV 
                phi_global_enemy = (turret_yaw_from_lidar + phi_offset + 360) % 360 
                best_matching_point = None; smallest_angular_difference = float('inf')
                for lidar_point in filtered_device_lidar_points:
                    angle_relative_to_body = lidar_point.get('angle', 0.0)
                    lidar_point_global_angle = (player_body_yaw + angle_relative_to_body + 360) % 360
                    angular_diff = (lidar_point_global_angle - phi_global_enemy + 180 + 360) % 360 - 180
                    if abs(angular_diff) < smallest_angular_difference:
                        smallest_angular_difference = abs(angular_diff); best_matching_point = lidar_point
                if best_matching_point:
                    dist = best_matching_point.get('distance')
                    if dist is not None:
                        enemy_distances.append({'phi': phi_global_enemy, 'distance': float(dist),
                                                'body_size': ENEMY_BODY_SIZE, 'turret_size': ENEMY_TURRET_SIZE })
                    else: print(f"Warning [/detect]: Matched LiDAR point (phi {phi_global_enemy:.2f}) has no distance.")
    elif not last_lidar_data : print("Warning [/detect]: last_lidar_data is None, LiDAR processing skipped.")
    last_enemy_data = {'timestamp': current_ts, 'enemies': enemy_distances}
    with log_lock, open('logs/enemy.json', 'a', encoding='utf-8') as f:
        json.dump(last_enemy_data, f, ensure_ascii=False); f.write('\n')
    if os.path.exists(image_path):
        try: os.remove(image_path)
        except Exception as e: print(f"Error: 임시 이미지 파일 {image_path} 삭제 실패: {e}")
    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    global last_lidar_data
    data = request.get_json(force=True)
    last_lidar_data = data
    # print(f"DEBUG [/info]: Received data. playerTurretX(Yaw): {last_lidar_data.get('playerTurretX')}, playerTurretY(Pitch): {last_lidar_data.get('playerTurretY')}")
    return jsonify({'status': 'success', 'control': ''})

@app.route('/get_action', methods=['POST'])
def get_action():
    global last_lidar_data, last_enemy_data 
    
    turret_yaw_current = 0.0    # 좌우 각도
    current_turret_pitch = 0.0  # 상하 각도
    turret_angles_source = "defaults" 

    if last_lidar_data:
        # --- 현재 포탑 상태 (last_lidar_data로부터 가져오기) ---
        # 사용자 확인: X가 Yaw(좌우), Y가 Pitch(상하)
        # CSV 헤더가 Player_Turret_X, Player_Turret_Y 이므로, JSON 키는 playerTurretX, playerTurretY 일 가능성이 높음.
        turret_yaw_current = last_lidar_data.get('playerTurretX', 0.0)   # <<--- X가 Yaw (좌우)
        current_turret_pitch = last_lidar_data.get('playerTurretY', 0.0) # <<--- Y가 Pitch (상하)
        turret_angles_source = "last_lidar_data (X:Yaw, Y:Pitch)"
        print(f"DEBUG [/get_action]: Using turret angles from last_lidar_data: Yaw(X)={turret_yaw_current:.2f}, Pitch(Y)={current_turret_pitch:.2f}")
    else:
        _data_from_post_for_fallback = request.get_json(force=True) 
        _turret_info_from_post = _data_from_post_for_fallback.get('turret', {})
        # 사용자 확인: POST 데이터 내 turret 객체에서 x가 Yaw, y가 Pitch
        turret_yaw_current = _turret_info_from_post.get('x', 0.0) # POST 데이터의 'x' (Yaw) 사용 (Fallback)
        current_turret_pitch = _turret_info_from_post.get('y', 0.0) # POST 데이터의 'y' (Pitch) 사용 (Fallback)
        turret_angles_source = "post_data_fallback (turret.x:Yaw, turret.y:Pitch)"
        print(f"Warning [/get_action]: last_lidar_data is None. Falling back to turret angles from POST data: Yaw(x)={turret_yaw_current:.2f}, Pitch(y)={current_turret_pitch:.2f}")

    cmd = {
        'moveWS': {'command': 'STOP', 'weight': 1.0}, 'moveAD': {'command': '', 'weight': 0.0},
        'turretQE': {'command': '', 'weight': 0.0}, 'turretRF': {'command': '', 'weight': 0.0},
        'fire': False
    }

    if last_enemy_data and last_enemy_data.get('enemies'):
        target_enemy = min(last_enemy_data['enemies'], key=lambda e: e.get('distance', float('inf')))
        phi_target_enemy = target_enemy['phi']
        dist_to_target = target_enemy['distance']
        calculated_target_pitch = calculate_target_pitch(dist_to_target)
        
        print(f"--- 조준 정보 (현재 각도 소스: {turret_angles_source}) ---")
        print(f"  적 전차 거리 (dist_to_target): {dist_to_target:.2f}")
        print(f"  계산된 목표 발사각 (calculated_target_pitch): {calculated_target_pitch:.2f} 도")
        print(f"  현재 포탑 좌우각 (turret_yaw_current): {turret_yaw_current:.2f} 도")
        print(f"  현재 포탑 상하각 (current_turret_pitch): {current_turret_pitch:.2f} 도")

        delta_yaw = ((phi_target_enemy - turret_yaw_current + 180) % 360) - 180
        yaw_aim_weight = min(abs(delta_yaw) / 180.0, 1.0)
        is_aimed_horizontally = abs(delta_yaw) <= FIRE_THRESHOLD_DEG

        delta_pitch = calculated_target_pitch - current_turret_pitch
        pitch_aim_weight = min(abs(delta_pitch) / PITCH_ADJUST_RANGE_FOR_WEIGHT, 1.0)
        is_aimed_vertically = abs(delta_pitch) <= PITCH_FIRE_THRESHOLD_DEG
        
        print(f"  수평각 차이 (delta_yaw): {delta_yaw:.2f} 도 (목표수평각: {phi_target_enemy:.2f})")
        print(f"  수직각 차이 (delta_pitch): {delta_pitch:.2f} 도")
        print(f"  수평 조준 완료: {is_aimed_horizontally}, 수직 조준 완료: {is_aimed_vertically}")

        if is_aimed_horizontally and is_aimed_vertically:
            cmd['fire'] = True; print(f"  명령: 발사 (FIRE!)")
        else:
            cmd['fire'] = False; current_commands_log = []
            if not is_aimed_horizontally:
                if delta_yaw > FIRE_THRESHOLD_DEG: cmd['turretQE'] = {'command': 'E', 'weight': yaw_aim_weight}; current_commands_log.append("Yaw: E")
                elif delta_yaw < -FIRE_THRESHOLD_DEG: cmd['turretQE'] = {'command': 'Q', 'weight': yaw_aim_weight}; current_commands_log.append("Yaw: Q")
            if not is_aimed_vertically:
                if delta_pitch > PITCH_FIRE_THRESHOLD_DEG: cmd['turretRF'] = {'command': 'R', 'weight': pitch_aim_weight}; current_commands_log.append("Pitch: R")
                elif delta_pitch < -PITCH_FIRE_THRESHOLD_DEG: cmd['turretRF'] = {'command': 'F', 'weight': pitch_aim_weight}; current_commands_log.append("Pitch: F")
            if current_commands_log: print(f"  명령: 조준 중 - {', '.join(current_commands_log)}")
            elif not (is_aimed_horizontally and is_aimed_vertically): print(f"  명령: 조준 중 (경계값 근처 또는 조준 명령 없음)")
    return jsonify(cmd)

# 나머지 라우트 함수들은 이전과 동일하게 유지됩니다.
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
        'trackingMode': False, 'detactMode': True, 'logMode': True,
        'enemyTracking': False, 'saveSnapshot': False, 'saveLog': True,
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
    app.run(host='0.0.0.0', port=5004, debug=True)
