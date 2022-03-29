
from django.contrib import admin
from django.urls import path
from django.conf.urls import include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('detect.urls')),
    #path('home/', include('home.urls')),

]
 