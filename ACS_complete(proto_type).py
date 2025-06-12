#깃에 같이 올려놓은 TESTMap 과 같이 사용하는걸 권장합니다.
# =============================================================================
# Flask 서버: 자율주행 → 정지 후 자율조준 통합 구현
# =============================================================================
from flask import Flask, request, jsonify
import os
import json
import re
import threading
import time
import math
import glob
import numpy as np
from queue import PriorityQueue
from ultralytics import YOLO

app = Flask(__name__)

# ----- 모델 설정 -----
model = YOLO('best.pt')  # 학습된 YOLO 모델

# ----- 탱크 크기 정보 -----
# 게임 내 플레이어와 적 탱크의 차체/포탑 크기를 정의합니다. (가로, 세로, 높이)
PLAYER_BODY_SIZE    = (3.667, 1.582, 8.066) # 우리 탱크의 차체 크기 (가로, 세로, 높이)
PLAYER_TURRET_SIZE = (3.297, 2.779, 5.891) # 우리 탱크의 포탑 크기
ENEMY_BODY_SIZE     = (3.303, 1.131, 6.339) # 적 탱크의 차체 크기
ENEMY_TURRET_SIZE   = (2.681, 3.094, 2.822) # 적 탱크의 포탑 크기

# ----- 전역 변수 및 동기화 설정 -----
last_lidar_data = None # /info 엔드포인트를 통해 업데이트됨 - 게임에서 받은 내 탱크의 최신 정보(위치, 포탑 각도 등)를 저장할 변수
last_enemy_data = None # /detect 엔드포인트에서 탐지된 적 정보를 저장하는 변수
log_lock = threading.Lock() # 여러 요청이 동시에 로그 파일에 쓰는 것을 방지

# ----- 설정값 -----
IMAGE_W = 2560 # 입력받는 이미지의 가로 해상도 (픽셀 수수)
HFOV    = 46.05 # 카메라의 수평 시야각 (Horizontal Field of View), 도 단위

# ----- 조준 관련 상수 -----
FIRE_THRESHOLD_DEG = 0.5 # 수평 발사 허용 오차 (도)
PITCH_FIRE_THRESHOLD_DEG = 0.1 # 수직오차각도 허용범위
PITCH_ADJUST_RANGE_FOR_WEIGHT = 30.0 # 상하속도 조절 weight
AIMING_YAW_OFFSET_DEG        = -0.5
PITCH_AIM_OFFSET_DEG          = 1.2

# ── 스캔 모드용 전역 변수 (90° 스텝) ──
SCAN_STEP_DEG   = 90.0   # 한 스텝당 회전량
PAUSE_SEC       = 1.0    # 목표 도달 후 멈춰 있을 시간 (초)
scan_origin_yaw = None
scan_index      = 0
pause_start     = None
scan_lap_count  = 0     # 완료된 회전 수 카운트

# ----- 자율주행 모듈 전역 설정 -----
GRID_SIZE = 300
maze = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]
DESTINATIONS = [(130, 180)]
current_dest_index = 0
TARGET_THRESHOLD = 30.0
ANGLE_THRESHOLD  = 0.1
FOV_DEG          = 70
DIST_THRESH      = 15
MAX_DIFF         = 30

device_yaw      = 0.0
previous_pos    = None
collision_count = 0
collision_lock  = threading.Lock()
goal_reached    = False
is_avoiding     = False

# ----- 발사각(Pitch) 계산 모델 계수(다항회귀 분석석) -----
# pitch ≈ c0*distance³ + c1*distance² + c2*distance + c3
PITCH_MODEL_COEFFS = [
    1.18614662e-05,  # c0 (distance³의 계수)
    -3.20931503e-03, # c1 (distance²의 계수)
    3.87703588e-01,  # c2 (distance의 계수)
    -11.55315302e+00  # c3 (상수항)
]
pitch_equation_model    = np.poly1d(PITCH_MODEL_COEFFS)

# ----------------------------------------------------------------------------
# 헬퍼 함수들
# ----------------------------------------------------------------------------

def world_to_grid(x: float, z: float) -> tuple:
    """
    세계 좌표 (x, z)를 그리드 인덱스 (i, j)로 변환.
    맵 범위를 벗어날 경우 경계값으로 클램프(clamp).
    """
    i = max(0, min(GRID_SIZE-1, int(x)))
    j = max(0, min(GRID_SIZE-1, int(z)))
    return i, j


def heuristic(a: tuple, b: tuple) -> float:
    """
    A* 알고리즘 휴리스틱 함수.
    여기서는 맨해튼 거리(수직+수평) 사용.
    f(n) = g(n) + h(n) 에서 h(n)에 해당.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(pos: tuple) -> list:
    """
    현재 셀(pos)에서 이동 가능한 이웃 셀(상/하/좌/우/대각선) 목록 반환.
    맵 경계 및 장애물(maze == 1) 검사 포함.
    """
    # 대각선 제외 4방향 탐색으로 수정 -> 8방향 설정시 경로 탐색시 오류 발생해서 이후 재적용 예정(8방향으로 수정완료, 2025_06_09))
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]
    neighbors = []
    for dx, dz in directions:
        nx, nz = pos[0] + dx, pos[1] + dz
        # 맵 내에 있고 장애물이 아니면 추가
        if 0 <= nx < GRID_SIZE and 0 <= nz < GRID_SIZE and maze[nx][nz] == 0:
            neighbors.append((nx, nz))
    return neighbors


class Node:
    """
    A* 탐색에서 사용하는 노드 객체.
    Attributes:
      position: (i, j) 격자 위치
      parent: 이전 노드 링크 (경로 추적용)
      g: 시작점에서 현재 노드까지 실제 비용
      h: 현재 노드에서 목표까지 추정 비용
      f: g + h (총 비용)
    """
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        # 우선순위 큐(PriorityQueue)에 넣을 때 f값으로 비교
        return self.f < other.f


def a_star(start: tuple, goal: tuple) -> list:
    """
    A* 경로 탐색 함수.
    Inputs:
      start: 출발 셀 (i, j)
      goal: 목표 셀 (i, j)
    Returns:
      경로 리스트 [(i1,j1), (i2,j2), ...] 순서대로.
      경로가 없으면 [start] 반환.
    """
    open_set = PriorityQueue()
    open_set.put((0, Node(start)))  # f=0, start 노드 삽입
    closed_set = set()

    while not open_set.empty():
        _, current = open_set.get()  # f가 가장 작은 노드 꺼내기
        # 목표에 도달했으면 경로 구성 후 반환
        if current.position == goal:
            path = []
            node = current
            while node:
                path.append(node.position)
                node = node.parent
            return path[::-1]  # 역순(출발→도착)

        closed_set.add(current.position)
        # 인접 노드 탐색
        for neighbor_pos in get_neighbors(current.position):
            if neighbor_pos in closed_set:
                continue

            dx = neighbor_pos[0] - current.position[0]
            dz = neighbor_pos[1] - current.position[1]
            step_cost = math.sqrt(dx * dx + dz * dz)  # 이동 거리 = 1 또는 √2
            
            # 새 노드 생성 및 비용 계산
            neighbor = Node(neighbor_pos, current)
            neighbor.g = current.g + step_cost # 기존의 고정값 1이 아닌 이동거리에 따른 가중치를 반영
            neighbor.h = heuristic(neighbor_pos, goal)
            neighbor.f = neighbor.g + neighbor.h
            open_set.put((neighbor.f, neighbor))

    # 경로를 찾지 못한 경우 시작 위치 반환
    return [start]


def calculate_angle(cur: tuple, nxt: tuple) -> float:
    """
    현재 셀(cur)에서 다음 셀(nxt)로 향하는 벡터의 yaw(방향) 각도 계산.
    반환값 범위: [0, 360)
    """
    dx = nxt[0] - cur[0]
    dz = nxt[1] - cur[1]
    angle = math.degrees(math.atan2(dz, dx))
    return (angle + 360) % 360


#전방에 장애물에 여부를 판단하는 함수 선언 2025_06_10 => 각도 범위값 변경(2025_06_11)
def obstacle_ahead(lidar_points, fov_deg=FOV_DEG, dist_thresh=DIST_THRESH):
    front_dists = []
    for p in lidar_points:
        angle_view = p.get('angle')
        if angle_view < 30 or angle_view > 330:
            if not p.get('isDetected') or p.get('verticalAngle') != 0:
                continue
            front_dists.append(p['distance'])
    return (min(front_dists) if front_dists else float('inf')) < dist_thresh


#전방에 장애물이 존재하면 거리 가중치에 근거해서 회피각도를 결정하는 함수 선언 2025_06_11
def compute_avoidance_direction_weighted(lidar_points, current_yaw, danger_dist=20.0, angle_delta=60):
    left_risk, right_risk = 0.0, 0.0
    left_count, right_count = 0, 0

    for p in lidar_points:
        if not p.get('isDetected') or p.get('verticalAngle') != 0:
            continue

        dist = p.get('distance', float('inf'))
        if dist > danger_dist or dist <= 0.5:
            continue

        angle = p.get('angle', 0.0) % 360
        risk = 1.0 / (dist + 1e-6)

        # 0~180도 = 오른쪽 / 180~360도 = 왼쪽
        if 0 <= angle <= 180:
            right_risk += risk
            right_count += 1
        else:
            left_risk += risk
            left_count += 1

    print(f"[DEBUG] 좌 포인트 수: {left_count}, 우 포인트 수: {right_count}")
    print(f"[DEBUG] 좌 위험도: {left_risk:.2f}, 우 위험도: {right_risk:.2f}")

    if left_risk > right_risk:
        return (current_yaw - angle_delta) % 360  # 오른쪽 회피
    else:
        return (current_yaw + angle_delta) % 360  # 왼쪽 회피


def calculate_target_pitch(distance):
    """거리에 따라 필요한 포탄의 발사각(Pitch)을 계산합니다."""
    min_gun_pitch = -5.0  # 실제 탱크의 최소 발사각
    max_gun_pitch = 9.75  # 실제 탱크의 최대 발사각
    
    # 다항식 모델을 사용하여 거리에 따른 초기 발사각을 계산산
    initial_calculated_pitch = pitch_equation_model(distance)
    
    # 계산된 발사각이 탱크의 실제 가동 범위를 벗어나지 않도록 값을 조정
    final_target_pitch = np.clip(initial_calculated_pitch, min_gun_pitch, max_gun_pitch)
    
   
    if abs(final_target_pitch - initial_calculated_pitch) > 0.01: 
        print(f"DEBUG (calculate_target_pitch): Distance: {distance:.2f}, Initial Calc Pitch: {initial_calculated_pitch:.2f}, Clamped Target Pitch: {final_target_pitch:.2f} (Limits: {min_gun_pitch:.2f} to {max_gun_pitch:.2f})")
        
    return final_target_pitch # 범위 이외의 값을 가질때 측정값을 넘긴다.


def _process_yolo_detection(image_file):
    """이미지를 받아 YOLO 탐지를 수행하고 결과를 반환하며, 임시 파일을 정리합니다."""
    image_path = 'temp_image.jpg'
    try:
        # 전송받은 이미지를 서버에 임시 파일로 저장
        image_file.save(image_path)
        # 저장된 이미지 파일로 YOLO 모델을 실행하여 객체 탐지를 수행
        results = model(image_path)
        # 탐지 결과를 numpy 배열로 변환
        yolo_detections = results[0].boxes.data.cpu().numpy()
        
        target_classes = {0: "tank", 1: "car"}
        filtered_results = []
        
        for box in yolo_detections:
            class_id = int(box[5])
            # 탐지된 객체가 우리가 원하는 클래스(tank, car)에 속하는지 확인
            if class_id in target_classes:
                # 결과 데이터를 리스트에 추가합니다.
                filtered_results.append({
                    'className': target_classes[class_id],
                    'bbox': [float(c) for c in box[:4]],
                    'confidence': float(box[4]),
                    'color': '#00FF00', 'filled': False, 'updateBoxWhileMoving': True
                })
        return filtered_results
    finally:
        # try 블록이 성공하든 실패하든 항상 임시 파일을 삭제합니다.
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception as e:
                # 파일 삭제에 실패하면 에러 메시지를 출력합니다.
                print(f"Error: 임시 이미지 파일 {image_path} 삭제 실패: {e}")

def _get_filtered_lidar_points():
    """전역 변수 last_lidar_data에서 직접 LiDAR 포인트를 가져와 필터링합니다."""
    global last_lidar_data
    # last_lidar_data가 없으면 빈 리스트를 반환
    if not last_lidar_data:
        return []
    raw_points = last_lidar_data.get('lidarPoints', [])
    # 포인트 데이터가 리스트 형태가 아니면 빈 리스트를 반환
    if not isinstance(raw_points, list):
        return []
    # 'lidarPoints' 리스트에서 수직각(verticalAngle)이 0.0이고, 실제로 감지된(isDetected) 포인트만 추출
    return [p for p in raw_points if p.get('verticalAngle') == 0.0 and p.get('isDetected')]


def _find_distance_for_detection(detection, lidar_points, state):
    """탐지된 객체 하나와 LiDAR 포인트 목록을 받아, 가장 일치하는 거리 값을 찾아 반환합니다."""
    # 1. 탐지된 객체의 절대 각도 계산
    x1, _, x2, _ = detection['bbox']
    u_center = (x1 + x2) / 2.0 # 바운딩박스 중앙을 잡음

    phi_offset = (u_center / IMAGE_W - 0.5) * HFOV # 정규화 한후에 각도값으로 변환

    phi_global_enemy = (state['turret_yaw'] + phi_offset + AIMING_YAW_OFFSET_DEG + 360) % 360 # 적게 움직이는 방향으로 만듦
    
    # 이 계산된 phi 값을 detection 딕셔너리에 추가하여 나중에 사용할 수 있게 합니다.
    detection['phi'] = phi_global_enemy
    
    # 2. 각도 차이가 가장 작은 LiDAR 포인트 찾기
    best_match = None
    smallest_angular_diff = float('inf') # 가장 작은 각도 차이를 저장할 변수

    # 필터링된 모든 LiDAR 포인트에 대해 반복
    for point in lidar_points:
        # LiDAR 포인트의 전역 각도를 계산
        lidar_global_angle = (state['turret_yaw'] + point.get('angle', 0.0) + 360) % 360
        # 각 LiDAR 포인트의 전역 각도와 탐지된 적의 전역 각도 사이의 차이를 계산(-180 ~ 180도 범위)
        angular_diff = (lidar_global_angle - phi_global_enemy + 180 + 360) % 360 - 180
        
        # 현재 포인트의 각도 차이가 이전에 찾은 최소 차이보다 작으면,
        if abs(angular_diff) < smallest_angular_diff:
            # 이 포인트를 '가장 일치하는' 포인트로 간주하고 정보를 업데이트
            smallest_angular_diff = abs(angular_diff)
            best_match = point
    # 3. 거리 값 반환
    # 가장 일치하는 LiDAR 포인트를 찾았다면,
    if best_match:
        # 해당 포인트의 거리(distance) 값을 반환
        return best_match.get('distance')
    return None


def _log_data(filepath, data):
    """주어진 데이터를 지정된 파일에 JSON 형태로 로그를 남깁니다."""
    with log_lock, open(filepath, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

# ----------------------------------------------------------------------------
# 주요 엔드포인트
# ----------------------------------------------------------------------------
@app.route('/init', methods=['GET'])
def init():
    return jsonify({
        'startMode':'start','blStartX':150,'blStartY':10,'blStartZ':10,
        'rdStartX':150,'rdStartY':10,'rdStartZ':290,
        'trackingMode':False,'detactMode':True,'logMode':True,
        'enemyTracking':False,'saveSnapshot':False,'saveLog':True,'saveLidarData':True,
        'lux':30000,
        'player_body_size':PLAYER_BODY_SIZE,'player_turret_size':PLAYER_TURRET_SIZE,
        'enemy_body_size':ENEMY_BODY_SIZE,'enemy_turret_size':ENEMY_TURRET_SIZE
    })

@app.route('/info', methods=['POST'])
def info():
    """시뮬레이터로부터 주기적으로 탱크의 상태 정보(LiDAR 포함)를 받아 전역 변수에 저장합니다."""
    global last_lidar_data
    data = request.get_json(force=True) or {}
    last_lidar_data = data
    print(f"DEBUG [/info]: Received data. playerTurretX(Yaw): {last_lidar_data.get('playerTurretX')}, playerTurretY(Pitch): {last_lidar_data.get('playerTurretY')}")
    return jsonify({'status': 'success', 'control': ''})

@app.route('/detect', methods=['POST'])
def detect():
    """메인 탐지 로직: 이미지와 LiDAR 데이터를 융합하여 적의 거리를 계산합니다."""
    # 전역 변수인 last_lidar_data와 last_enemy_data를 함수 내에서 수정할 수 있도록 선언
    global last_lidar_data, last_enemy_data
    # --- 1. 이미지 및 현재 상태 가져오기 ---
    image_file = request.files.get('image')
    if not image_file:
        # 이미지가 없으면 에러 메시지를 반환합니다.
        return jsonify({"error": "No image received"}), 400

    # 현재 시간과 탱크의 포탑(turret), 차체(body)의 방향(yaw) 정보를 저장할 딕셔너리를 생성
    current_state = {
        'time': time.time(),
        'turret_yaw': 0.0,
        'body_yaw': 0.0
    }
    # 만약 이전에 수신된 LiDAR 데이터(/info 엔드포인트를 통해)가 있다면, 그 정보로 현재 상태를 업데이트
    if last_lidar_data:
        current_state['time'] = last_lidar_data.get('time', current_state['time']) 
        current_state['turret_yaw'] = last_lidar_data.get('playerTurretX', 0.0)  #playerTurretX는 /info에서 가져오는 변수
        current_state['body_yaw'] = last_lidar_data.get('playerBodyX', 0.0) #playerBodyX는 /info에서 가져오는 변수

    # --- 2. 역할별 함수 호출로 작업 수행 ---
    # (1) 이미지에서 객체 탐지 
    yolo_detections = _process_yolo_detection(image_file)
    # 탐지된 객체 정보를 로그 파일에 기록
    _log_data('logs/detections.json', {'timestamp': current_state['time'], 'turretYaw': current_state['turret_yaw'], 'detections': yolo_detections})

    # (2) 최신 LiDAR 포인트 필터링
    lidar_points = _get_filtered_lidar_points()

    # (3) 탐지된 탱크와 LiDAR 포인트를 융합하여 거리 계산
    enemy_distances = []
        # 탐지된 각 객체에 대해 반복
    for det in yolo_detections:
        # 탐지된 객체가 'tank'일 경우에만 거리 계산을 수행
        if det['className'] == 'tank': #class가 0인것 tank인것만 거리를 계산
            # 헬퍼 함수를 호출하여 해당 탱크까지의 거리를 찾습니다.
            distance = _find_distance_for_detection(det, lidar_points, current_state)
            # 거리가 성공적으로 계산되었을 경우, 적 정보 목록에 추가
            if distance is None: distance = 50.0
            enemy_distances.append({
                    'phi': det['phi'], # _find_distance_for_detection에서 계산된 phi 추가
                    'distance': distance,
                    'body_size': ENEMY_BODY_SIZE,
                    'turret_size': ENEMY_TURRET_SIZE
                })

    # --- 3. 최종 결과 기록 및 반환 ---
    # 최종적으로 계산된 적 정보를 전역 변수에 저장하여 다른 함수(/get_action)에서 사용할 수 있게 합니다.
    last_enemy_data = {'timestamp': current_state['time'], 'enemies': enemy_distances}
    # 적 정보를 로그 파일에 기록합니다.
    _log_data('logs/enemy.json', last_enemy_data)

    # 탐지 결과를 JSON 형태로 클라이언트에게 반환
    return jsonify(yolo_detections)

@app.route('/get_action', methods=['POST'])
def get_action():
    global device_yaw, previous_pos, goal_reached, current_dest_index, is_avoiding
    global scan_origin_yaw, scan_index, pause_start, scan_lap_count
    data=request.get_json(force=True) or {}
    pos=data.get('position',{})
    x=float(pos.get('x',0))
    z=float(pos.get('z',0))
    lidar_points=last_lidar_data.get('lidarPoints',[]) if isinstance(last_lidar_data,dict) else []
    
    # 변수 초기화
    turret_yaw_current = 0.0   # 현재 포탑의 좌우 각도
    current_turret_pitch = 0.0 # 현재 포탑의 상하 각도
    turret_angles_source = "defaults" # 각도 정보의 출처를 기록하기 위한 변수

    # 1) 자율주행 단계
    if not goal_reached:
        dest_x, dest_z = DESTINATIONS[current_dest_index]
        dist_to_goal = math.hypot(x-dest_x, z-dest_z)

        if dist_to_goal < TARGET_THRESHOLD:
            print(f"[INFO] 목적지 {current_dest_index} 도달: ({dest_x},{dest_z})")
            # 다음 목적지로 이동
            if current_dest_index < len(DESTINATIONS)-1:
                current_dest_index += 1
                print(f"[INFO] 다음 목적지 인덱스: {current_dest_index}")
            else:
                print(f"[INFO] 목표 도달: 거리 {dist_to_goal:.2f}m → 정지 명령 전송")
                print(f"총 충돌 횟수는 {collision_count}번 입니다.")
                goal_reached = True
            return jsonify({
                'moveWS':{'command':'','weight':0.0}, 'moveAD':{'command':'','weight':0.0},
                'turretQE':{'command':'','weight':0.0}, 'turretRF':{'command':'','weight':0.0}, 'fire':False
            })
        else:
            print("아직 전차가 목표지점으로 이동중입니다.")
        goal_reached = False
        if previous_pos:
            dx, dz = x - previous_pos[0], z - previous_pos[1]
            movement = math.hypot(dx, dz)
            if movement > 0.2:  # ← 임계값 증가 (예: 0.01 → 0.2)
                device_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
        previous_pos = (x, z)
        start_cell = world_to_grid(x, z)
        goal_cell = world_to_grid(dest_x, dest_z)
        path = a_star(start_cell, goal_cell)
        next_cell = path[1] if len(path) > 1 else start_cell

        target_yaw = calculate_angle(start_cell, next_cell)
        diff=(target_yaw-device_yaw+360)%360
        if diff>180: diff-=360
        if obstacle_ahead(lidar_points):
            avoid_yaw=compute_avoidance_direction_weighted(lidar_points,device_yaw)
            diff=(avoid_yaw-device_yaw+360)%360
            if diff>180: diff-=360
            move_ad={'command':'A' if diff>0 else 'D','weight':min(abs(diff)/MAX_DIFF,1.0)}
        else:
            move_ad={'command':'','weight':0.0} if abs(diff)<ANGLE_THRESHOLD else {'command':'A' if diff>0 else 'D','weight':0.2}
        return jsonify({'moveWS':{'command':'W','weight':0.3},'moveAD':move_ad,'turretQE':{'command':'' ,'weight':0.0},'turretRF':{'command':'','weight':0.0},'fire':False})
    # /info 엔드포인트에서 받은 `last_lidar_data`가 있는지 확인하여 포탑 각도 정보를 가져옵니다.
    if last_lidar_data:
        turret_yaw_current = last_lidar_data.get('playerTurretX', 0.0)   # X가 Yaw (좌우)
        current_turret_pitch = last_lidar_data.get('playerTurretY', 0.0) # Y가 Pitch (상하)
        print(f"DEBUG [/get_action]: Using turret angles from last_lidar_data: Yaw(X)={turret_yaw_current:.2f}, Pitch(Y)={current_turret_pitch:.2f}")
    else:
        # 만약 `last_lidar_data`가 없다면,/get_action 요청에 직접 포함된 데이터에서 포탑 정보를 가져옵니다.
        fallback = request.get_json(force=True).get('turret', {})
        # 사용자 확인: POST 데이터 내 turret 객체에서 x가 Yaw, y가 Pitch
        turret_yaw_current = fallback.get('x', 0.0) # POST 데이터의 'x' (Yaw) 사용 (Fallback)
        current_turret_pitch = fallback.get('y', 0.0) # POST 데이터의 'y' (Pitch) 사용 (Fallback)
        

    
    cmd = {
        'moveWS': {'command': 'STOP', 'weight': 1.0}, 'moveAD': {'command': '', 'weight': 0.0},
        'turretQE': {'command': '', 'weight': 0.0}, 'turretRF': {'command': '', 'weight': 0.0},
        'fire': False
    }

    # 만약 탐지된 적(/detect의 결과)이 있다면 조준 및 발사 로직을 실행합니다.
    if last_enemy_data and last_enemy_data.get('enemies'):
        # 스캔 초기화
        scan_origin_yaw = None
        pause_start = None
        scan_lap_count = 0
 
        #search_passed_halfway = False # 탐색상태 초기화 
        # 여러 적 중에서 거리가 가장 가까운 적을 목표로 선택합니다.
        target_enemy = min(last_enemy_data['enemies'], key=lambda e: e.get('distance', float('inf')))
        phi_target_enemy = target_enemy['phi']     # 목표 적의 수평 각도
        dist_to_target = target_enemy['distance']  # 목표 적까지의 거리
        
        # 거리에 따른 목표 발사각을 계산합니다.
        calculated_target_pitch = calculate_target_pitch(dist_to_target) + PITCH_AIM_OFFSET_DEG
        
        print(f"--- 조준 정보 (현재 각도 소스: {turret_angles_source}) ---")
        print(f"  적 전차 거리 (dist_to_target): {dist_to_target:.2f}")
        print(f"  계산된 목표 발사각 (calculated_target_pitch): {calculated_target_pitch:.2f} 도")
        print(f"  현재 포탑 좌우각 (turret_yaw_current): {turret_yaw_current:.2f} 도")
        print(f"  현재 포탑 상하각 (current_turret_pitch): {current_turret_pitch:.2f} 도")

        # 수평 조준: 목표 좌우 각도와 현재 포탑 좌우 각도의 차이를 계산합니다.
        delta_yaw = ((phi_target_enemy - turret_yaw_current + 180) % 360) - 180 #좌우 조절 거리
        # 각도 차이가 클수록 포탑 회전 속도를 빠르게 하기 위한 가중치(weight)를 계산합니다.
        yaw_aim_weight = min(abs(delta_yaw) / 180.0, 1.0)  # 가로조절 속도 weight값 - 현재 180으로 나누어 계산
        # 좌우 각도 차이가 설정된 오차 범위 내에 들어왔는지 확인하여 수평 조준 완료 여부를 판단합니다.

        # 수직 조준: 목표 상하 각도와 현재 포탑 상하 각도의 차이를 계산합니다.
        delta_pitch = calculated_target_pitch - current_turret_pitch
        # 각도 차이에 비례하여 포탑 상하 이동 속도 가중치를 계산합니다.
        pitch_aim_weight = min(abs(delta_pitch) / PITCH_ADJUST_RANGE_FOR_WEIGHT, 1.0)
        # 상하 각도 차이가 설정된 오차 범위 내에 들어왔는지 확인하여 수직 조준 완료 여부를 판단합니다.
        #is_aimed_vertically = abs(delta_pitch) <= PITCH_FIRE_THRESHOLD_DEG
        
        # 조준 속도를 더 빠르게 하기 위해 계산된 가중치에 상수를 곱해 증폭시킵니다.
        YAW_WEIGHT_BOOST = 5
        PITCH_WEIGHT_BOOST = 5
        yaw_aim_weight = min(yaw_aim_weight * YAW_WEIGHT_BOOST, 1.0)
        pitch_aim_weight = min(pitch_aim_weight * PITCH_WEIGHT_BOOST, 1.0)

        print(f"  수평각 차이 (delta_yaw): {delta_yaw:.2f} 도 (목표수평각: {phi_target_enemy:.2f})")
        print(f"  수직각 차이 (delta_pitch): {delta_pitch:.2f} 도")
        

        if abs(delta_yaw) <= FIRE_THRESHOLD_DEG and abs(delta_pitch) <= PITCH_FIRE_THRESHOLD_DEG:
            cmd['fire'] = True
        else:
            if abs(delta_yaw) > FIRE_THRESHOLD_DEG:
                cmd['turretQE'] = {'command': 'E' if delta_yaw > 0 else 'Q', 'weight': yaw_aim_weight}
            if abs(delta_pitch) > PITCH_FIRE_THRESHOLD_DEG:
                cmd['turretRF'] = {'command': 'R' if delta_pitch > 0 else 'F', 'weight': pitch_aim_weight}
    else:
        # ── 최초 스캔 모드 진입 시 초기화 (한 번만) ──
        if scan_origin_yaw is None:
            scan_origin_yaw = turret_yaw_current
            scan_index = 0
            pause_start = None
            scan_lap_count = 0

        # 목표 스캔 각도 계산
        target_yaw = (scan_origin_yaw + SCAN_STEP_DEG * scan_index) % 360
        delta = ((target_yaw - turret_yaw_current + 180) % 360) - 180

        # 회전 중: 가중치 기반 속도 제어
        if abs(delta) > 1.0:
            pause_start = None
            speed_weight = min(abs(delta) / 30.0, 0.5)
            cmd['turretQE'] = {'command': 'E' if delta > 0 else 'Q', 'weight': speed_weight}
        else:
            # 목표 각도 도달: 멈춤 및 스텝/랩 카운트 증가 처리
            if pause_start is None:
                pause_start = time.time()
            if time.time() - pause_start < PAUSE_SEC:
                cmd['turretQE'] = {'command': '', 'weight': 0.0}
            else:
                scan_index += 1
                if scan_index >= int(360 / SCAN_STEP_DEG):
                    scan_index = 0
                    scan_lap_count += 1
                if scan_lap_count >= 2:
                    cmd['turretQE'] = {'command': '', 'weight': 0.0}
                else:
                    pause_start = None

    return jsonify(cmd)

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data=request.get_json(force=True) or {}
    dst=data.get('destination')
    if dst:
        x,y,z=map(float,dst.split(','))
        return jsonify({'status':'ok','destination':{'x':x,'y':y,'z':z}})
    return jsonify({'status':'error','message':'Missing destination'}),400
@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data=request.get_json(force=True) or {}
    obs=data.get('obstacles',[])
    if isinstance(obs,dict): obs=[obs]
    for o in obs:
        if isinstance(o,(list,tuple)):
            x,z=o[0],o[1]
        else:
            x,z=o.get('x'),o.get('z')
        i,j=world_to_grid(float(x),float(z))
        maze[i][j]=1
    return jsonify({'status':'ok'})

@app.route('/collision', methods=['POST'])
def collision():
    global collision_count
    with collision_lock:
        collision_count+=1
        cnt=collision_count
    return jsonify({'status':'ok','collision_count':cnt})

@app.route('/collision/count', methods=['GET'])
def get_collision_count():
    with collision_lock:
        return jsonify({'collision_count':collision_count})

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    return jsonify({'status':'OK','message':'Bullet impact data received'})

@app.route('/get_map', methods=['GET'])
def get_map():
    obstacles=[{'x':i,'z':j} for i in range(GRID_SIZE) for j in range(GRID_SIZE) if maze[i][j]==1]
    return jsonify({'obstacles':obstacles})

@app.route('/start', methods=['GET'])
def start():
    return jsonify({'control':''})

# ----------------------------------------------------------------------------
# 서버 실행
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
