# =============================================================================
# Flask 서버 + A* 기반 자율주행 및 필수 API 통합 구현
# =============================================================================
from flask import Flask, request, jsonify
from queue import PriorityQueue
import math
import os
import torch
from ultralytics import YOLO
import json
import re #라이더 데이터를 새로운 포멧으로 읽어오기 위한 모듈 import(2025_06_09)
import threading

app = Flask(__name__)
# YOLO 객체 탐지 모델 로드 (Ultralytics YOLOv8)
model = YOLO('best.pt')  # best.pt: 학습된 모델 파일

# ----------------------------------------------------------------------------
# 전역 설정
# ----------------------------------------------------------------------------

GRID_SIZE = 300  # 2D 격자 맵 크기 (NxN)

# 실제 맵 => 0: 이동 가능, 1: 장애물
maze = [[0]*GRID_SIZE for _ in range(GRID_SIZE)]

current_dest_index = 0             # 현재 목표 인덱스(2025_06_09)
TARGET_THRESHOLD = 20.0           # 목표 도달로 간주할 거리 임계값

# 전방 장애물 판단여부에 활용할 변수 선언(2025_06_10)
ANGLE_THRESHOLD = 0.1  # diff가 이 값 이하일 때 회전 무시
FOV_DEG = 70       # 전방 몇 도 내 장애물 판단
DIST_THRESH = 15   # 장애물로 판단할 거리 (m)
MAX_DIFF = 30

# 순차 경로 탐색할 목적지 목록(2025_06_09)
DESTINATIONS = [
    (130, 180)
]

device_yaw = 0.0    # 전차 현재 방향(도 단위)
previous_pos = None  # 마지막 위치 저장 (x, z)

goal_reached = False # 전차의 목적지 도달여부를 판단하기 위한 전역변수

is_avoiding = False  # 회피 중 여부(2025_06_11)

#라이더 센서 데이터를 json파일에 대한 I/O 접근이 아닌 직접 읽어오기 위한 변수(2025_06_09)
last_lidar_data = []

# 전역 충돌 카운터와 락 2025_06_11
collision_count = 0
collision_lock = threading.Lock()

# ----------------------------------------------------------------------------
# 헬퍼 함수들
# ----------------------------------------------------------------------------
#전방에 장애물에 여부를 판단하는 함수 선언 2025_06_10 => 각도 범위값 변경(2025_06_11)
def obstacle_ahead(lidar_points, fov_deg=FOV_DEG, dist_thresh=DIST_THRESH):
    front_dists = []
    for p in lidar_points:
        angle_view = p.get('angle')
        if angle_view < 20 or angle_view > 340:
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

# ----------------------------------------------------------------------------
# 주요 엔드포인트: 시뮬레이터 연동
# ----------------------------------------------------------------------------
@app.route('/init', methods=['GET'])
def init():
    """
    시뮬레이터 최초 초기화 API.
    시작 위치, 모드 설정 등을 JSON으로 반환.
    """
    return jsonify({
        "startMode": "start",
        # 블록 좌표
        "blStartX": 150, "blStartY": 10, "blStartZ": 10,
        # 레드 좌표
        "rdStartX": 150, "rdStartY": 10, "rdStartZ": 290,
        # 모드 설정
        "trackingMode": False,
        "detactMode": True,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    })

@app.route('/detect', methods=['POST'])
def detect():
    """
    이미지 파일을 받아 YOLOv8으로 객체 탐지 수행.
    Returns:
      리스트 of {class, bbox, confidence}
    """
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    # 임시 파일로 저장 후 모델로 추론
    temp_path = 'temp.jpg'
    image.save(temp_path)
    results = model.predict(temp_path)
    dets = results[0].boxes.data.cpu().numpy()

    target_classes = {0: 'car', 1: 'tank'}
    output = []
    for box in dets:
        cid = int(box[5])
        if cid not in target_classes:
            continue
        x1, y1, x2, y2 = box[:4]
        output.append({
            'class': target_classes[cid],
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(box[4]),
            'color': "#0400FF",
            'filled': True,
            'updateBoxWhileMoving': True
        })

    # 임시 이미지 제거
    os.remove(temp_path)
    return jsonify(output)

@app.route('/info', methods=['POST'])
def info():
    """
    LiDAR 및 기타 상태 데이터를 실시간으로 수신.
    실제 로직 필요 시 전역 변수 업데이트.
    """
    #실시간으로 라이더 데이터를 수신하기 위해서 last_lidar_data 선언(2025_06_09)
    global last_lidar_data
    data = request.get_json(force=True) or {}
    #전체 info에서 lidar data 값만 취사 선택해서 리스트로 저장(2025_06_09)
    points = data.get('lidarPoints', [])
    last_lidar_data = points 
    return jsonify({"status": "ok"})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    """
    장애물 좌표 목록을 받아 맵(maze) 업데이트.
    입력 포맷: [{'x':i, 'z':j}, ...] or [[i,j], ...]
    """
    data = request.get_json(force=True) or {}
    obs = data.get('obstacles', [])
    if isinstance(obs, dict):
        obs = [obs]
    for o in obs:
        if isinstance(o, (list, tuple)):
            x, z = o[0], o[1]
        else:
            x, z = o.get('x'), o.get('z')
        gi, gj = world_to_grid(float(x), float(z))
        maze[gi][gj] = 1  # 장애물 표시
    return jsonify({'status': 'ok'})

@app.route('/get_map', methods=['GET'])
def get_map():
    """
    현재 설정된 장애물 좌표 리스트 반환.
    """
    obstacles = [{'x': i, 'z': j} for i in range(GRID_SIZE) for j in range(GRID_SIZE) if maze[i][j] == 1]
    return jsonify({'obstacles': obstacles})

@app.route('/get_action', methods=['POST'])
def get_action():
    """
    자율 주행 명령 생성 API.
    입력: 현재 위치 JSON {'position': {'x':..., 'z':...}}
    처리:
      1) 목표 도달 확인
      2) 현재 yaw 업데이트 (이전 위치 기반)
      3) A* 경로 탐색 → 다음 셀 계산
      4) 방향 차이 계산 → 이동 명령 생성
    반환: 이동/회전/사격 명령 JSON
    """
    global previous_pos, device_yaw, goal_reached, current_dest_index #전역 설정한 변수 추가(current_dest_index 추가 2025_06_09)
    #실시간으로 라이더 데이터를 수신하기 위해서 last_lidar_data 선언(2025_06_09)
    global last_lidar_data

    data = request.get_json(force=True) or {}
    pos = data.get('position', {})
    x, z = float(pos.get('x', 0)), float(pos.get('z', 0))
    # LiDAR 포인트 리스트 형태로 last_lidar_data 사용
    lidar_points = last_lidar_data

    # 장애물 감지 여부(2025_06_11)
    obstacle_detected = obstacle_ahead(lidar_points)
        
    # 서버 측에서 도달 이후 goal_reached = True 상태를 저장하고,
    # 이후에는 전혀 명령을 주지 않게(또는 상태 고정) 처리
    # 현재 목표 설정

    # DESTINATIONS에 저장되어 있는 목적지를 순차적으로 탐색(2025_06_09)
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

    # 2) yaw 업데이트 (이전 위치가 있다면)
    # 현재 움직임이 매우 작으면 yaw 업데이트 생략(2025_06_11)
    if previous_pos:
        dx, dz = x - previous_pos[0], z - previous_pos[1]
        movement = math.hypot(dx, dz)
        if movement > 0.2:  # ← 임계값 증가 (예: 0.01 → 0.2)
            device_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
    previous_pos = (x, z)

    # A* 경로 탐색
    start_cell = world_to_grid(x, z)
    goal_cell = world_to_grid(dest_x, dest_z)
    path = a_star(start_cell, goal_cell)
    print(f"[DEBUG] A* path: {path[:5]}")  #← 경로가 제대로 잡히는지 확인
    next_cell = path[1] if len(path) > 1 else start_cell
    
    # 4) 방향 차이(diff) 계산
    target_yaw = calculate_angle(start_cell, next_cell)
    diff = (target_yaw - device_yaw + 360) % 360
    if diff > 180:
        diff -= 360
    print("계산 각도: ", (abs(diff) / 180))

    print("is_avoiding :", is_avoiding)
    # 5) 장애물 vs 경로 회전 분기 구간 설정(2025_06_10)
    #바뀐 변수로 교체 2025_06_11
    
    
    if obstacle_detected:
        # 장애물 회피용 목표 각도
        avoid_yaw = compute_avoidance_direction_weighted(lidar_points, device_yaw)
        diff = ((avoid_yaw - device_yaw + 360) % 360)
        print(f"현재 DIFF 값은 {diff} 입니다.")
        if diff > 180:
            diff -= 360
        print(f"장애물을 탐지했습니다. 회피가 필요한 구간입니다.")
        move_ad_cmd = {
            'command': 'A' if diff > 0 else 'D',
            'weight': min(abs(diff) / MAX_DIFF, 1.0)
        }

    else:
        if abs(diff) < ANGLE_THRESHOLD:
            move_ad_cmd = {'command': '', 'weight': 0.0}
        else:
            move_ad_cmd = {
                'command': 'A' if diff > 0 else 'D',
                'weight': 0.2
            }

    cmd = {
        'moveWS': {'command': 'W', 'weight': 0.3},
        'moveAD': move_ad_cmd,
        'turretQE': {'command': '', 'weight': 0.0},
        'turretRF': {'command': '', 'weight': 0.0},
        'fire': False
    }
    return jsonify(cmd)

@app.route('/set_destination', methods=['POST'])
def set_destination():
    """
    동적 목적지 설정 API.
    입력: {'destination': 'x,y,z'} 형식
    """
    data = request.get_json(force=True) or {}
    dst_str = data.get('destination')
    if dst_str:
        x, y, z = map(float, dst_str.split(','))
        return jsonify({'status': 'ok', 'destination': {'x': x, 'y': y, 'z': z}})
    return jsonify({'status': 'error', 'message': 'Missing destination'}), 400

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    """발사체 정보 수신 (미사용)"""
    return jsonify({'status': 'ok'})
#충돌 횟수를 카운팅하기 위한 collision count 세팅
@app.route('/collision', methods=['POST'])
def collision():
    """충돌 정보 수신 및 카운트 증가"""
    global collision_count

    # (Optional) POST로 넘어온 충돌 상세 정보가 필요하면 파싱
    # data = request.get_json()

    # 스레드 안전하게 카운트 증가
    with collision_lock:
        collision_count += 1
        current_count = collision_count

    return jsonify({
        'status': 'ok',
        'collision_count': current_count
    })

@app.route('/collision/count', methods=['GET'])
def get_collision_count():
    """현재까지 누적된 충돌 횟수 조회"""
    with collision_lock:
        return jsonify({'collision_count': collision_count})

@app.route('/start', methods=['GET'])
def start():
    """시뮬레이터 제어 시작 신호"""
    return jsonify({'control': ''})

# ----------------------------------------------------------------------------
# 서버 실행
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
