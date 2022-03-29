from django.urls import path

from . import viewsdl
from . import viewsml



urlpatterns = [
    path('detectML', viewsml.detectML, name='detectML'),
    path('detectDL', viewsdl.detectDL, name='detectDL'),
    path('home/', viewsml.home, name='home'),
    path('ML/', viewsml.ML, name='ML'),    
    path('DL/', viewsdl.DL, name='DL'),

]
