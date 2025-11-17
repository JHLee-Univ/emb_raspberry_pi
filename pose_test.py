import cv2
import mediapipe as mp
import numpy as np # 각도 계산을 위해 추가

# --- 1. MediaPipe 솔루션 초기화 ---
mp_drawing = mp.solutions.drawing_utils  # 랜드마크를 그리는 유틸리티
mp_pose = mp.solutions.pose            # Pose 추정 모델

# --- 2. 헬퍼 함수: 세 점 사이의 각도 계산 ---
def calculate_angle(a, b, c):
    """세 점 a, b, c가 주어졌을 때 b 지점의 각도를 계산합니다. (a-b-c 각도)"""
    # 좌표를 numpy 배열로 변환
    a = np.array(a) # 첫 번째 점 (예: 어깨)
    b = np.array(b) # 중간 점 (예: 팔꿈치)
    c = np.array(c) # 세 번째 점 (예: 손목)
    
    # 벡터 계산
    v1 = a - b
    v2 = c - b
    
    # 내적을 이용해 라디안(radian) 계산
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    # 0으로 나누기 방지
    if norm_v1 == 0 or norm_v2 == 0:
        return 0 

    cosine_angle = dot_product / (norm_v1 * norm_v2)
    
    # acos의 입력 범위는 [-1, 1]이므로 clamp
    if cosine_angle > 1.0:
        cosine_angle = 1.0
    elif cosine_angle < -1.0:
        cosine_angle = -1.0
        
    angle_rad = np.arccos(cosine_angle)
    
    # 라디안을 도로 변환
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# --- 3. 웹캠 열기 ---
# CV2 Video Capter (실시간 영상 받아오기)
cap = cv2.VideoCapture(0) # 0번 카메라 (노트북/데스크탑 기본 웹캠)

# MediaPipe Pose 모델 로드
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("카메라를 열 수 없습니다.")
            continue

        # (A) 전처리: BGR 이미지를 RGB로 변환 (MediaPipe는 RGB 입력 사용)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # 성능 향상을 위해 이미지 쓰기 방지
        
        # (B) 자세 추정 (뼈대 추출)
        results = pose.process(image_rgb)
        
        # (C) 후처리: 다시 BGR로 변환 (OpenCV 표시는 BGR 사용)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # (D) 랜드마크 및 연결선 그리기 (CV2 Circle, CV2 Line)
        if results.pose_landmarks:
            # mp_drawing.draw_landmarks가 내부적으로 cv2.circle, cv2.line을 사용합니다.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,          # 검출된 랜드마크
                mp_pose.POSE_CONNECTIONS,        # 랜드마크 연결선
                # 랜드마크(점) 스타일
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                # 연결선 스타일
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  
            )
            
            # --- [응용] 자세 "판단"을 위한 특징 추출 (예: 왼쪽 팔꿈치 각도) ---
            try:
                # 랜드마크 좌표 가져오기
                landmarks = results.pose_landmarks.landmark
                
                # 왼쪽 어깨, 팔꿈치, 손목 좌표
                shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow_l = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist_l = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                # 각도 계산
                angle_l = calculate_angle(shoulder_l, elbow_l, wrist_l)
                
                # 화면에 각도 표시 (랜드마크 좌표는 0~1 정규화 값이므로 실제 픽셀 좌표로 변환)
                h, w, _ = image.shape
                elbow_pixel_coord = tuple(np.multiply(elbow_l, [w, h]).astype(int))
                
                cv2.putText(image, f"L-Elbow: {int(angle_l)}", 
                            elbow_pixel_coord, # 텍스트 표시 위치
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # --- [확장] Scikit-learn을 이용한 자세 "판단" (간단한 예시) ---
                #
                # Scikit-learn을 사용하려면, 먼저 [angle_l, angle_r, ...] 등 
                # 여러 특징(feature) 벡터를 수집하고 'sitting', 'standing' 등
                # 라벨을 붙여 모델(예: SVM, KNN)을 *미리 학습*시켜야 합니다.
                #
                # if model.predict([[angle_l, ...]]) == 'sitting':
                #     cv2.putText(image, "SITTING", ...)
                #
                # (간단한 임계값 기반 판단 예시)
                if angle_l > 160:
                    pose_status = "L-Arm Straight"
                elif angle_l < 90:
                    pose_status = "L-Arm Bent"
                else:
                    pose_status = "L-Arm Normal"
                
                # 상태 표시줄
                cv2.rectangle(image, (0,0), (250, 60), (245,117,66), -1) # 배경
                cv2.putText(image, "STATUS", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, pose_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                # 관절이 화면 밖에 있거나 안보일 때 예외가 발생할 수 있음
                pass
            # --- 응용 끝 ---

        # (E) 화면 출력 (IM Show)
        # 좌우 반전(거울 모드)을 해서 보여주는 것이 자연스럽습니다.
        cv2.imshow('Pose Estimation (Press ESC to exit)', cv2.flip(image, 1))

        # 'ESC' 키를 누르면 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

# --- 4. 자원 해제 ---
cap.release()
cv2.destroyAllWindows()