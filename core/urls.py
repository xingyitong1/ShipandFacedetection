from django.contrib import admin
from django.urls import path, include  # add this


urlpatterns = [
    path('admin/', admin.site.urls),          
    path("auth/", include("authentication.urls")), # Auth routes - login / register
    path("", include("book.urls")),
    path('api/', include('Api.urls'))
]

handler400 = 'book.views.bad_request'
handler403 = 'book.views.permission_denied'
handler404 = 'book.views.page_not_found'
handler500 = 'book.views.server_error'
