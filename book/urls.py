from django.urls import path
from .views import HomeView


urlpatterns = [

    # HomePage
    path("",HomeView.as_view(), name='home'),



]



