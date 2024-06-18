import dlib
import cv2
import numpy as np
import math
import os
from scipy.spatial import distance as dist
from imutils import face_utils, resize
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

# 加载dlib模型
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("path_to_shape_predictor_68_face_landmarks.dat")

# 世界坐标系(UVW)
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745], # 34右眉左角
                         [-6.825897, 6.760612, 4.402142], # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414], # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654], # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652], # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531], # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],# 39嘴右上角
                         [0.000000, -3.116408, 6.097667], # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])# 6下巴角

# 相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

# 常数定义
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3

# 初始化计数器
COUNTER = 0
TOTAL = 0
mCOUNTER = 0
mTOTAL = 0
hCOUNTER = 0
hTOTAL = 0

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    return reprojectdst, euler_angle

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])
    B = np.linalg.norm(mouth[4] - mouth[7])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def process_video(video_path):
    global COUNTER, TOTAL, mCOUNTER, mTOTAL, hCOUNTER, hTOTAL
    cap = cv2.VideoCapture(video_path)
    results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[36:42]
            rightEye = shape[42:48]
            mouth = shape[48:68]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            mar = mouth_aspect_ratio(mouth)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            if mar > MAR_THRESH:
                mCOUNTER += 1
            else:
                if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                    mTOTAL += 1
                mCOUNTER = 0

            reprojectdst, euler_angle = get_head_pose(shape)
            har = euler_angle[0, 0]

            if har > HAR_THRESH:
                hCOUNTER += 1
            else:
                if hCOUNTER >= NOD_AR_CONSEC_FRAMES:
                    hTOTAL += 1
                hCOUNTER = 0

            if TOTAL >= 50 or mTOTAL >= 15 or hTOTAL >= 15:
                results.append("Fatigue detected")
            else:
                results.append("No fatigue detected")

    cap.release()
    return results

@csrf_exempt
def detect_fatigue(request):
    if request.method == 'POST':
        if 'video' not in request.FILES:
            return JsonResponse({"error": "No video uploaded"}, status=400)

        file = request.FILES['video']
        video_path = './temp_video.mp4'
        with open(video_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        results = process_video(video_path)
        os.remove(video_path)
        return JsonResponse({"results": results})

    return JsonResponse({"error": "Invalid request"}, status=400)

def page_not_found(request, exception):
    context = {}
    response = render(request, "errors/404.html", context=context)
    response.status_code = 404
    return response

def server_error(request):
    context = {}
    response = render(request, "errors/500.html", context=context)
    response.status_code = 500
    return response
