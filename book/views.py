from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin
from util.useful import get_n_days_ago,create_clean_dir,change_col_format

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

TODAY=get_n_days_ago(0,"%Y%m%d")
PAGINATOR_NUMBER = 5
allowed_models = ['Category','Publisher','Book','Member','UserActivity']


# HomePage

class HomeView(LoginRequiredMixin,TemplateView):
    login_url = 'login'
    template_name = "index.html"
    context={}

  
    users = User.objects.all()
    for user in users:
        print(user.get_username(),user.is_superuser)

    def get(self,request, *args, **kwargs):

        # book_count = Book.objects.aggregate(Sum('quantity'))['quantity__sum']
        #
        # data_count = {"book":book_count,
        #             "member":Member.objects.all().count(),
        #             "category":Category.objects.all().count(),
        #             "publisher":Publisher.objects.all().count(),}
        #
        # user_activities= UserActivity.objects.order_by("-created_at")[:5]
        # user_avatar = { e.created_by:Profile.objects.get(user__username=e.created_by).profile_pic.url for e in user_activities}
        # short_inventory =Book.objects.order_by('quantity')[:5]
        #
        # current_week = date.today().isocalendar()[1]
        # new_members = Member.objects.order_by('-created_at')[:5]
        # new_members_thisweek = Member.objects.filter(created_at__week=current_week).count()
        # lent_books_thisweek = BorrowRecord.objects.filter(created_at__week=current_week).count()
        #
        # books_return_thisweek = BorrowRecord.objects.filter(end_day__week=current_week)
        # number_books_return_thisweek = books_return_thisweek.count()
        # new_closed_records = BorrowRecord.objects.filter(open_or_close=1).order_by('-closed_at')[:5]
        #
        # self.context['data_count']=data_count
        # self.context['recent_user_activities']=user_activities
        # self.context['user_avatar']=user_avatar
        # self.context['short_inventory']=short_inventory
        # self.context['new_members']=new_members
        # self.context['new_members_thisweek']=new_members_thisweek
        # self.context['lent_books_thisweek']=lent_books_thisweek
        # self.context['books_return_thisweek']=books_return_thisweek
        # self.context['number_books_return_thisweek']=number_books_return_thisweek
        # self.context['new_closed_records']=new_closed_records
 
        return render(request, self.template_name, self.context)
    
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


# def process_video(video_path, model, opt, device, half):
#     cap = cv2.VideoCapture(video_path)
#     out_path = os.path.join('media/results', 'processed_' + os.path.basename(video_path))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     ret, frame = cap.read()
#     vw = frame.shape[1]
#     vh = frame.shape[0]
#     output_video = cv2.VideoWriter(out_path, fourcc, 20.0, (vw, vh))
#
#     while True:
#         grabbed, image = cap.read()
#         if not grabbed:
#             break
#
#         img = cv2.resize(image, (850, 500))
#         img0 = img.copy()
#         img = letterbox(img0, new_shape=opt.img_size)[0]
#         img = np.array(img)
#         img = img[:, :, ::-1].transpose(2, 0, 1)
#         img = np.ascontiguousarray(img)
#
#         img = torch.from_numpy(img).to(device)
#         img = img.half() if half else img.float()
#         img /= 255.0
#         if img.ndimension() == 3:
#             img = img.unsqueeze(0)
#
#         pred = model(img, augment=False)[0]
#         pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
#
#         det = pred[0]
#         if det is not None and len(det):
#             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
#             for *xyxy, conf, cls in reversed(det):
#                 label = f'{model.names[int(cls)]} {conf:.2f}'
#                 plot_one_box(img0, xyxy, label=label, color=[random.randint(0, 255) for _ in range(3)])
#
#         image = cv2.resize(img0, (vw, vh))
#         output_video.write(image)
#
#     cap.release()
#     output_video.release()
#     return out_path

def upload_image(request):
    # if request.method == 'POST' and request.FILES.get('file'):
    #     uploaded_file = request.FILES['file']
    #     fs = FileSystemStorage()
    #     file_path = fs.save(uploaded_file.name, uploaded_file)
    #     file_path = fs.path(file_path)
    #
    #     opt = parse_opt()
    #     device = select_device(opt.device)
    #     half = device.type != 'cpu'
    #     model = attempt_load(opt.weights, map_location=device)
    #     if half:
    #         model.half()
    #
    #     img = cv2.imread(file_path)
    #     img = cv2.resize(img, (opt.img_size, opt.img_size))
    #     img0 = img.copy()
    #     img = letterbox(img, new_shape=opt.img_size)[0]
    #     img = np.stack(img, 0)
    #     img = img[:, :, ::-1].transpose(2, 0, 1)
    #     img = np.ascontiguousarray(img)
    #
    #     img = torch.from_numpy(img).to(device)
    #     img = img.half() if half else img.float()
    #     img /= 255.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #
    #     t1 = time_synchronized()
    #     pred = model(img, augment=False)[0]
    #     pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    #     t2 = time_synchronized()
    #
    #     for det in pred:
    #         if det is not None and len(det):
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    #             for *xyxy, conf, cls in reversed(det):
    #                 label = f'{model.names[int(cls)]} {conf:.2f}'
    #                 plot_one_box(img0, xyxy, label=label, color=[random.randint(0, 255) for _ in range(3)])
    #
    #     file_name = "processsed_" + uploaded_file.name
    #     result_path = os.path.join('media/', file_name)
    #     cv2.imwrite(result_path, img0)
    #
    #     return JsonResponse({'processed_image_url': fs.url(file_name)})
    #
    # return JsonResponse({'error': 'Invalid request'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)
def upload_video(request):
    # print("upload_video called")
    # if request.method == 'POST' and request.FILES.get('file'):
    #     uploaded_file = request.FILES['file']
    #     upload_fs = FileSystemStorage(location='media')
    #     uploaded_file_path = upload_fs.save(uploaded_file.name, uploaded_file)
    #     uploaded_file_path = upload_fs.path(uploaded_file_path)
    #
    #     opt = parse_opt()
    #     device = select_device(opt.device)
    #     half = device.type != 'cpu'
    #     model = attempt_load(opt.weights, map_location=device)
    #     if half:
    #         model.half()
    #
    #     result_path = process_video(uploaded_file_path, model, opt, device, half)
    #     print(f"Processed video saved to: {result_path}")
    #     return JsonResponse({'processed_video_url': upload_fs.url(os.path.basename(result_path))})
    #
    # return JsonResponse({'error': 'Invalid request'}, status=400)
    return JsonResponse({'error': 'Invalid request'}, status=400)

def driver(request):
    return render(request, "includes/driver_warning.html")


def upload_image_driver(request):


    return JsonResponse({'error': 'Invalid request'}, status=400)