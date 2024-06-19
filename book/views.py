from django.core.files.base import ContentFile
from django.shortcuts import render
from django.views.generic import TemplateView
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
import os
import cv2
import torch
import random
import numpy as np
import argparse

from imutils import face_utils
from Yolov5.models.experimental import attempt_load
from Yolov5.utils.datasets import letterbox
from Yolov5.utils.general import non_max_suppression, scale_coords
from Yolov5.utils.torch_utils import select_device, time_synchronized
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import subprocess
from util.useful import get_n_days_ago, create_clean_dir, change_col_format
import dlib
from scipy.spatial import distance as dist
from PIL import Image,ImageDraw,ImageFont
import sys
sys.path.append('./Yolov5')

TODAY = get_n_days_ago(0, "%Y%m%d")
PAGINATOR_NUMBER = 5
allowed_models = ['Category', 'Publisher', 'Book', 'Member', 'UserActivity']

# models的参数配置
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./Yolov5/weights/ship-best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args(args=[])
    return opt


def plot_one_box(img, x, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Define the fatigue detection related constants and functions

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
MAR_THRESH = 0.5
MOUTH_AR_CONSEC_FRAMES = 3
HAR_THRESH = 0.3
NOD_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
mCOUNTER = 0
mTOTAL = 0
hCOUNTER = 0
hTOTAL = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./Yolov5/weights/shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

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
def put_chinese_text(img, text, position, color=(255, 0, 0), font_size=20):
    # Convert the image to PIL format
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype('E:/ShipandFacedetection/book/STZHONGS.TTF', font_size)
    draw.text(position, text, font=font, fill=color)
    # Convert back to OpenCV format
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img
def process_fatigue_frame(frame):
    global COUNTER, TOTAL, mCOUNTER, mTOTAL, hCOUNTER, hTOTAL
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        
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
        
        # cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "COUNTER: {}".format(COUNTER), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "Blinks: {}".format(TOTAL), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        # cv2.putText(frame, "Yawning: {}".format(mTOTAL), (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # cv2.putText(frame, "mCOUNTER: {}".format(mCOUNTER), (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        frame = put_chinese_text(frame, "人脸数: {}".format(len(rects)), (10, 30))
        frame = put_chinese_text(frame, "眨眼计数: {}".format(COUNTER), (10, 60))
        frame = put_chinese_text(frame, "眼睛比例: {:.2f}".format(ear), (10, 90))
        frame = put_chinese_text(frame, "眨眼次数: {}".format(TOTAL), (10, 120))
        frame = put_chinese_text(frame, "打哈欠次数: {}".format(mTOTAL), (10, 150))
        frame = put_chinese_text(frame, "嘴巴比例: {:.2f}".format(mar), (10, 180))
        frame = put_chinese_text(frame, "打哈欠计数: {}".format(mCOUNTER), (10, 210))

        if TOTAL >= 3 or mTOTAL >= 3:
            frame = put_chinese_text(frame, "别睡了!!!", (100, 250), color=(255, 0, 0), font_size=60)
    return frame

# HomePage

class HomeView(LoginRequiredMixin, TemplateView):
    login_url = 'login'
    template_name = "index.html"
    context = {}

    users = User.objects.all()
    for user in users:
        print(user.get_username(), user.is_superuser)

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.context)


def process_video(video_path, model, opt, device, half):
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join('media', 'processed_' + os.path.splitext(os.path.basename(video_path))[0] + '.mp4')

    # Change the fourcc to 'avc1' for H.264 encoding
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    ret, frame = cap.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    output_video = cv2.VideoWriter(out_path, fourcc, 20.0, (vw, vh))

    while True:
        grabbed, image = cap.read()
        if not grabbed:
            break

        img = cv2.resize(image, (850, 500))
        img0 = img.copy()
        img = letterbox(img0, new_shape=opt.img_size)[0]
        img = np.array(img)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        det = pred[0]
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                label = f'{model.names[int(cls)]} {conf:.2f}'
                plot_one_box(img0, xyxy, label=label, color=[random.randint(0, 255) for _ in range(3)])

        image = cv2.resize(img0, (vw, vh))
        output_video.write(image)

    cap.release()
    output_video.release()
    return out_path


# 处理上传的图片
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        file_path = fs.path(file_path)

        opt = parse_opt()
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(opt.weights, map_location=device)
        if half:
            model.half()

        img = cv2.imread(file_path)
        img = cv2.resize(img, (opt.img_size, opt.img_size))
        img0 = img.copy()
        img = letterbox(img, new_shape=opt.img_size)[0]
        img = np.stack(img, 0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    plot_one_box(img0,xyxy, label=label, color=[random.randint(0, 255) for _ in range(3)])

        file_name = "processsed_" + uploaded_file.name
        result_path = os.path.join('media/', file_name)
        cv2.imwrite(result_path, img0)

        return JsonResponse({'processed_image_url': fs.url(file_name)})

    return JsonResponse({'error': 'Invalid request'}, status=400)


def upload_video(request):
    print("upload_video called")
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        upload_fs = FileSystemStorage(location='media')
        uploaded_file_path = upload_fs.save(uploaded_file.name, uploaded_file)
        uploaded_file_path = upload_fs.path(uploaded_file_path)

        opt = parse_opt()
        device = select_device(opt.device)
        half = device.type != 'cpu'
        model = attempt_load(opt.weights, map_location=device)
        if half:
            model.half()

        result_path = process_video(uploaded_file_path, model, opt, device, half)
        print(f"Processed video saved to: {result_path}")
        return JsonResponse({'processed_video_url': upload_fs.url(os.path.basename(result_path))})

    return JsonResponse({'error': 'Invalid request'}, status=400)

# def capture_fatigue_video(request):
#     if request.method == 'POST':
#         cap = cv2.VideoCapture(0)  # 打开默认摄像头
#         if not cap.isOpened():
#             return JsonResponse({'error': 'Unable to access the camera'}, status=400)
#
#         out_path = os.path.join('media', 'fatigue_live_processed.mp4')
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')
#         ret, frame = cap.read()
#         vw = frame.shape[1]
#         vh = frame.shape[0]
#         output_video = cv2.VideoWriter(out_path, fourcc, 20.0, (vw, vh))
#
#         while True:
#             grabbed, frame = cap.read()
#             if not grabbed:
#                 break
#
#             frame = process_fatigue_frame(frame)
#             output_video.write(frame)
#
#             # 按下 'q' 键退出循环
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#
#         cap.release()
#         output_video.release()
#         cv2.destroyAllWindows()
#         return JsonResponse({'processed_video_url': 'media/fatigue_live_processed.mp4'})

    # 使用外置摄像头捕捉视频并进行疲劳监测
def capture_fatigue_video(output_path):
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out_path = os.path.join(output_path, 'fatigue_capture.mp4')
    ret, frame = cap.read()
    vw = frame.shape[1]
    vh = frame.shape[0]
    output_video = cv2.VideoWriter(out_path, fourcc, 20.0, (vw, vh))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_fatigue_frame(frame)
        output_video.write(frame)

        # Display the frame
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    return out_path


def upload_fatigue_video(request):
    if request.method == 'POST' and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        upload_fs = FileSystemStorage(location='media')
        uploaded_file_path = upload_fs.save(uploaded_file.name, uploaded_file)

        uploaded_file_path = upload_fs.path(uploaded_file_path)

        cap = cv2.VideoCapture(uploaded_file_path)
        out_path = os.path.join('media/', 'fatigue_processed_' + os.path.basename(uploaded_file_path))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        ret, frame = cap.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        output_video = cv2.VideoWriter(out_path, fourcc, 20.0, (vw, vh))

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame = process_fatigue_frame(frame)
            output_video.write(frame)

        cap.release()
        output_video.release()
        return JsonResponse({'processed_video_url': upload_fs.url(os.path.basename(out_path))})

    return JsonResponse({'error': 'Invalid request'}, status=400)


def convert_to_mp4(webm_file_path):
    upload_fs = FileSystemStorage(location='media')
    mp4_file_path = webm_file_path.replace('.webm', '.mp4')
    full_webm_file_path = upload_fs.path(webm_file_path)
    full_mp4_file_path = upload_fs.path(mp4_file_path)

    command = [
        'ffmpeg',
        '-i', full_webm_file_path,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        full_mp4_file_path
    ]

    subprocess.run(command, check=True)
    return mp4_file_path

def upload_real_time_video(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']
        # 存储视频片段到服务器
        upload_fs = FileSystemStorage(location='media')
        file_path = upload_fs.save(video_file.name, video_file)
        uploaded_file_path = upload_fs.path(file_path)

        cap = cv2.VideoCapture(uploaded_file_path)
        out_path = os.path.join('media/', 'fatigue_real_time_processed_' + os.path.basename(uploaded_file_path).replace('.webm', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        ret, frame = cap.read()
        vw = frame.shape[1]
        vh = frame.shape[0]
        output_video = cv2.VideoWriter(out_path, fourcc, 20.0, (vw, vh))

        while True:
            grabbed, frame = cap.read()
            if not grabbed:
                break

            frame = process_fatigue_frame(frame)
            output_video.write(frame)

        cap.release()
        output_video.release()
        return JsonResponse({'processed_video_url': upload_fs.url(os.path.basename(out_path))})


    return JsonResponse({'error': 'Invalid request'}, status=400)

def driver(request):
    return render(request,"includes/driver_warning.html")

def monitor(request):
    return render(request,"includes/monitor.html")

# Handle Errors

def page_not_found(request, exception):
    context = {}
    response = render(request, "errors/404.html", context=context)
    response.status_code = 404
    return response


def server_error(request, exception=None):
    context = {}
    response = render(request, "errors/500.html", context=context)
    response.status_code = 500
    return response


def permission_denied(request, exception=None):
    context = {}
    response = render(request, "errors/403.html", context=context)
    response.status_code = 403
    return response


def bad_request(request, exception=None):
    context = {}
    response = render(request, "errors/400.html", context=context)
    response.status_code = 400
