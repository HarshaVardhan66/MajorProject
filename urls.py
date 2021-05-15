from django.urls import path
from . import views

app_name = 'facedetect'

urlpatterns = [
    # homepage/
    path('', views.homepage, name='homepage'),
    #services/
    path('services/', views.services, name='services'),
    # services/upload/
    path('services/upload/', views.upload, name='upload'),
    # train/
    path('services/train/', views.train, name='train'),
    # recognise/
    path('services/recognise/', views.recognise, name='recognise'),

    #upload/livecapture/
    path('services/upload/livecapture/', views.live_capture, name='LiveCapture'),

    #upload/submitvideo/
    path('services/upload/submitvideo/', views.video_submit, name='VideoSubmit'),

    # train/training
    path('services/train/training', views.trainingresult, name='training'),

    #howto/
    path('howto/', views.howto, name='howto'),


]