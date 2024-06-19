from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from .views import HomeView, upload_image, upload_video,driver, upload_fatigue_video, monitor,upload_real_time_video

urlpatterns = [
    # HomePage
    path("", HomeView.as_view(), name='home'),
    # 将处理视频的视图函数与 URL 路由关联起来
    path("upload_image/", upload_image, name="upload_image"),
    path("upload_video/", upload_video, name="upload_video"),
    path('driver_warning/', driver, name='driver_warning'),
    path('monitor/', monitor, name='monitor'),
    path("upload_fatigue_video/", upload_fatigue_video, name="upload_fatigue_video"),
    path("upload_real_time_video/", upload_real_time_video, name="upload_real_time_video"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)