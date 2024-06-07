from django.urls import include, path
from rest_framework import routers
from Api import views
from rest_framework.urlpatterns import format_suffix_patterns



urlpatterns = [

    #Home
    path('', views.apiOverview, name="api-overview"),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),
]


urlpatterns = format_suffix_patterns(urlpatterns)
