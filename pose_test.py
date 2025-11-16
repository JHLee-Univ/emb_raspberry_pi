import cv2
import mediapipe as mp

# MediaPipe Pose 솔루션 및 그리기 유틸리티 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 웹캠 (기본 카메라 0번) 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("오류: 카메라를 열 수 없습니다.")
    exit()

print("카메라가 성공적으로 열렸습니다. 'q' 키를 누르면 종료됩니다.")

while cap.isOpened():
    # 1. 실시간 영상 받아오기 (cv2.VideoCapture)
    success, image = cap.read()
    if not success:
        print("카메라 프레임을 읽는 데 실패했습니다.")
        continue

    # 성능 향상을 위해 이미지를 '쓰기 불가'로 설정 (선택 사항)
    image.flags.writeable = False
    
    # BGR 이미지를 RGB로 변환 (MediaPipe는 RGB 입력을 사용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. MediaPipe로 뼈대 추출 (자세 추정)
    results = pose.process(image_rgb)
    
    # 이미지를 다시 '쓰기 가능'으로 설정
    image.flags.writeable = True
    
    # BGR 이미지에 뼈대 그리기
    # 3. 뼈대 그리기 (cv2.circle, cv2.line 역할)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,  # 뼈대의 연결선
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
        
        # --- '자세 판단' 로직이 들어갈 위치 ---
        # 
        # '자세 판단'(예: "서있음", "앉았음", "손들기")을 하려면
        # 'results.pose_landmarks'에서 랜드마크 좌표(x, y, z)를 추출하여
        # Scikit-learn, TensorFlow 등으로 미리 학습된 분류 모델에 입력해야 합니다.
        #
        # 지금은 단순히 "Detected"라는 텍스트를 띄웁니다.
        cv2.putText(image, "Pose Detected", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # (참고) 특정 랜드마크(예: 코)의 좌표 얻기:
        # nose_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        # print(f"코 좌표: x={nose_landmark.x}, y={nose_landmark.y}")


    # 4. 분석 결과 화면 출력 (cv2.imshow)
    cv2.imshow('MediaPipe Pose Estimation (Press "q" to quit)', image)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
pose.close()

print("프로그램을 종료합니다.")