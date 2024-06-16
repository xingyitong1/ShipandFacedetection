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
from Yolov5.models.experimental import attempt_load
from Yolov5.utils.datasets import letterbox
from Yolov5.utils.general import non_max_suppression, scale_coords,plot_one_box
from Yolov5.utils.general import non_max_suppression, scale_coords
from Yolov5.utils.torch_utils import select_device, time_synchronized
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin
from util.useful import get_n_days_ago, create_clean_dir, change_col_format

#
import sys
sys.path.append('E:/ShipandFacedetection/Yolov5')

sys.path.append('E:/ShipandFacedetection/Yolov5')

TODAY = get_n_days_ago(0, "%Y%m%d")
PAGINATOR_NUMBER = 5
allowed_models = ['Category', 'Publisher', 'Book', 'Member', 'UserActivity']


# models的参数配置
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./Yolov5/weights/ship-best.pt', help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='./Yolov5/weights/ship-best.pt',
                        help='model.pt path(s)')
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
    out_path = os.path.join('media/results', 'processed_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                    plot_one_box(xyxy, img0, label=label, color=[random.randint(0, 255) for _ in range(3)])
                    plot_one_box(img0,xyxy, label=label, color=[random.randint(0, 255) for _ in range(3)])

        file_name = "processsed_"+uploaded_file.name
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
    return response

def driver(request):
    return render(request, "includes/driver_warning.html")


def upload_image_driver(request):


    return JsonResponse({'error': 'Invalid request'}, status=400)