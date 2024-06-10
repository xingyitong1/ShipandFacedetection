from django.urls import path
from .views import HomeView,upload_image,upload_video
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    # HomePage
    path("",HomeView.as_view(), name='home'),
    path("upload_image/",upload_image,name = "upload_image"),
    path("upload_video",upload_video,name = "upload_video"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


