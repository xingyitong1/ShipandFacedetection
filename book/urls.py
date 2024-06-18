from django.urls import path
from .views import HomeView, upload_image, upload_video, driver, upload_fatigue_video

urlpatterns = [
    # HomePage
    path("", HomeView.as_view(), name='home'),
    # 将处理视频的视图函数与 URL 路由关联起来
    path("upload_image/", upload_image, name="upload_image"),
    path("upload_video/", upload_video, name="upload_video"),
    path('driver_warning/', driver, name='driver_warning'),
    path("upload_fatigue_video/", upload_fatigue_video, name="upload_fatigue_video"),
]
