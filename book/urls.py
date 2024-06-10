from django.urls import path
from .views import HomeView,upload_image
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    # HomePage
    path("",HomeView.as_view(), name='home'),
    path("upload_image/",upload_image,name = "upload_image"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


