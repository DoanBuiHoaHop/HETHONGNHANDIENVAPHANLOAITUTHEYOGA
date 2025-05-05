import streamlit as st
st.set_page_config(page_title="Yoga Pose Detection", layout="wide")
import time
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import tempfile
import os
import mediapipe as mp

# Danh sách các tư thế yoga
yoga_poses = [
    "Bound Angle Pose", "Chair Pose", "Dancer Pose", "Downward_dog",
    "Half Moon Pose", "Tree Pose", "Triangle", "Warrior Pose"
]

# Load mô hình
@st.cache_resource
def load_yoga_model():
    return tf.keras.models.load_model('./yoga-model.h5')

yoga_model = load_yoga_model()

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Tính góc giữa ba điểm
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Phân tích tư thế từ landmark
def analyze_pose(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose_detector.process(img_rgb)
    if not result.pose_landmarks:
        return {}

    h, w, _ = img.shape
    lm = result.pose_landmarks.landmark

    def get(idx): return int(lm[idx].x * w), int(lm[idx].y * h)

    try:
        # Lấy tọa độ các khớp
        shoulder_left = get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        hip_left = get(mp_pose.PoseLandmark.LEFT_HIP.value)
        knee_left = get(mp_pose.PoseLandmark.LEFT_KNEE.value)
        ankle_left = get(mp_pose.PoseLandmark.LEFT_ANKLE.value)
        wrist_left = get(mp_pose.PoseLandmark.LEFT_WRIST.value)

        shoulder_right = get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        hip_right = get(mp_pose.PoseLandmark.RIGHT_HIP.value)
        knee_right = get(mp_pose.PoseLandmark.RIGHT_KNEE.value)
        ankle_right = get(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        wrist_right = get(mp_pose.PoseLandmark.RIGHT_WRIST.value)

        return {
            "knee_angle_left": calculate_angle(hip_left, knee_left, ankle_left),
            "spine_angle_left": calculate_angle(shoulder_left, hip_left, (hip_left[0], hip_left[1] + 100)),
            "torso_angle_left": calculate_angle(shoulder_left, hip_left, ankle_left),
            "left_arm_angle": calculate_angle(hip_left, shoulder_left, wrist_left),
            "knee_angle_right": calculate_angle(hip_right, knee_right, ankle_right),
            "spine_angle_right": calculate_angle(shoulder_right, hip_right, (hip_right[0], hip_right[1] + 100)),
            "torso_angle_right": calculate_angle(shoulder_right, hip_right, ankle_right),
            "right_arm_angle": calculate_angle(hip_right, shoulder_right, wrist_right)
        }
    except:
        return {}

# Đưa ra phản hồi theo tư thế
def get_pose_feedback(pose_name, prob, pose_analysis):
    fb = []
    knee_left = pose_analysis.get("knee_angle_left", 0)
    knee_right = pose_analysis.get("knee_angle_right", 0)
    spine_left = pose_analysis.get("spine_angle_left", 0)
    spine_right = pose_analysis.get("spine_angle_right", 0)
    torso_left = pose_analysis.get("torso_angle_left", 0)
    torso_right = pose_analysis.get("torso_angle_right", 0)
    left_arm = pose_analysis.get("left_arm_angle", 0)
    right_arm = pose_analysis.get("right_arm_angle", 0)

    # Tree Pose
    if pose_name == "Tree Pose":
        feedback_given = False  
        if knee_left < 90 or knee_right < 90:
            fb.append("Great stability in Tree Pose!")
            feedback_given = True
        if not feedback_given:
            fb.append("Keep your knee slightly bent to maintain balance.")

    # Dancer Pose
    elif pose_name == "Dancer Pose":
        feedback_given = False
        if knee_left < 100 and knee_right < 100:
            fb.append("Bend your knee slightly")
            feedback_given = True
        if (torso_left + torso_right) / 2 < 90:
            fb.append("Ensure your torso is not leaning too far forward.")
            feedback_given = True
        if not feedback_given:
            fb.append("Great balance and form in Dancer Pose!")

    # Downward Dog
    elif pose_name == "Downward_dog":
        feedback_given = False
        arm_avg = (left_arm + right_arm) / 2
        torso_avg = (torso_left + torso_right) / 2

        if arm_avg < 160:
            fb.append("Try to straighten your arms more in Downward Dog.")
            feedback_given = True
        if torso_avg > 100:
            fb.append("Push your hips upward to form an inverted V shape.")
            feedback_given = True
        if not feedback_given:
            fb.append("Beautiful Downward Dog!.")
        
        # Chair Pose
    elif pose_name == "Chair Pose":
        feedback_given = False
        spine_avg = (spine_left + spine_right) / 2
        knee_avg = (knee_left + knee_right) / 2
        arm_avg = (left_arm + right_arm) / 2

        if knee_avg < 80:
            fb.append("Try bending your knees a bit more to deepen the Chair Pose.")
            feedback_given = True
        if spine_avg < 160:
            fb.append("Keep your back straighter in Chair Pose.")
            feedback_given = True
        if arm_avg < 150:
            fb.append("Lift your arms higher to align them with your ears.")
            feedback_given = True
        if not feedback_given:
            fb.append("Great posture and strength in Chair Pose!")

    # Triangle Pose
    elif pose_name == "Triangle":
        feedback_given = False
        spine_avg = (spine_left + spine_right) / 2
        arm_avg = (left_arm + right_arm) / 2

        if spine_avg < 90:
            fb.append("Longer torso, straighter spine.")
            feedback_given = True
        if arm_avg < 90:
            fb.append("Straighten your arms.")
            feedback_given = True
        if not feedback_given:
            fb.append("Excellent alignment in Triangle Pose!")

    # Warrior Pose
    elif pose_name == "Warrior Pose":
        feedback_given = False
        knee_avg = (knee_left + knee_right) / 2
        arm_avg = (left_arm + right_arm) / 2
        spine_avg = (spine_left + spine_right) / 2

        if knee_avg < 80 :
            fb.append("Bend front knee deeper.")
            feedback_given = True
        if arm_avg < 80 or arm_avg > 100:
            fb.append("Lift arms to shoulder height and extend")
            feedback_given = True
        if spine_avg < 170:
            fb.append("Keep your upper body upright and stable.")
            feedback_given = True
        if not feedback_given:
            fb.append("Strong and stable Warrior Pose!")

        # Half Moon Pose
    elif pose_name == "Half Moon Pose":
        feedback_given = False
        spine_avg = (spine_left + spine_right) / 2
        knee_avg = (knee_left + knee_right) / 2
        arm_avg = (left_arm + right_arm) / 2

        if torso_left < 60 or torso_right < 60 :
            fb.append("Lift your upper body more to avoid collapsing forward.")
            feedback_given = True
        if knee_avg < 160:
            fb.append("Slightly bend your standing knee to improve balance.")
            feedback_given = True
        if arm_avg < 90 or arm_avg > 110:
            fb.append("Keep your lifted arm aligned with your shoulder.")
            feedback_given = True
        if not feedback_given:
            fb.append("Beautiful Half Moon Pose! Excellent alignment and control.")
        # Bound Angle Pose
    elif pose_name == "Bound Angle Pose":
        feedback_given = False
        knee_avg = (knee_left + knee_right) / 2
        spine_avg = (spine_left + spine_right) / 2

        if knee_avg > 100:
            fb.append("Lower your knees.")
            feedback_given = True
        if spine_avg < 160:
            fb.append("Straighten your back.")
            feedback_given = True
        if not feedback_given:
            fb.append("Nice posture!")


    return fb

# Dự đoán tư thế
def imgProcess(image):
    img = cv2.resize(image, (224, 224)) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = yoga_model.predict(img, verbose=0)
    max_prob, class_idx = np.max(pred), np.argmax(pred)
    pose_name = yoga_poses[class_idx]
    analysis = analyze_pose(image)
    return pose_name, max_prob, analysis

# Vẽ keypoints và skeleton
def draw_keypoints_and_skeleton(img, landmarks, offset=(0, 0), size=None):
    h, w, _ = size if size else img.shape
    offset_x, offset_y = offset
    connections = mp.solutions.pose.POSE_CONNECTIONS
    for start, end in connections:
        x1, y1 = int(landmarks[start].x * w + offset_x), int(landmarks[start].y * h + offset_y)
        x2, y2 = int(landmarks[end].x * w + offset_x), int(landmarks[end].y * h + offset_y)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for lm in landmarks:
        x, y = int(lm.x * w + offset_x), int(lm.y * h + offset_y)
        cv2.circle(img, (x, y), 4, (0, 255, 255), -1)

# Hàm xử lý video
def run_yoga_detection_live(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tempfile.gettempdir(), "processed_output.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps // frame_skip, (width, height))

    frame_placeholder = st.empty()
    frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose_detector.process(img_rgb)

        if result.pose_landmarks:
            # Vẽ keypoints và khung xương
            draw_keypoints_and_skeleton(img, result.pose_landmarks.landmark)

            # Dự đoán tư thế và phân tích khớp
            pose_name, prob, analysis = imgProcess(img)

            # Hiển thị tên tư thế + độ chính xác
            label = f"{pose_name} ({prob*100:.1f}%)"
            pose_color = (0, 255, 0) if prob >= 0.6 else (0, 0, 255)
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pose_color, 2)

            # Lấy phản hồi từ kết quả phân tích tư thế
            feedback = get_pose_feedback(pose_name, prob, analysis)

            # Hiển thị phản hồi trên khung hình
            for i, line in enumerate(feedback):
                y = 60 + i * 30
                cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Cập nhật khung hình lên Streamlit
        frame_placeholder.image(img, channels="BGR", use_container_width=True)

        # Ghi video đầu ra
        out.write(img)

    cap.release()
    out.release()
    return output_path


import tempfile

st.title("Yoga Pose Recognition & Feedback")
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")  

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_file.read())
        temp_path = temp_file.name

    with st.spinner("Processing video..."):
        output_path = run_yoga_detection_live(temp_path)

    st.success("Video processed!")

    # Thêm nút tải video đã xử lý
    with open(output_path, "rb") as file:
        st.download_button(
            label="Download processed video",
            data=file,
            file_name="processed_output.mp4",
            mime="video/mp4"
        )

    # Chỉnh kích thước video hiển thị bằng HTML
    video_html = f'''
    <video width="800" controls>
        <source src="{output_path}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    '''
    st.markdown(video_html, unsafe_allow_html=True)
   
#python -m streamlit run app.py