
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('face_detector/', include('face_detector.urls')),
    path('admin/', admin.site.urls),
]
