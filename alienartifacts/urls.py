from django.urls import path
from . import views
from django.conf.urls.static import static # new
from django.conf import settings # new

app_name = 'alienartifacts'

urlpatterns = [
    #Example urls
    #Task-specific urls
    path("", views.index, name="index"),
    path("welcome", views.welcome, name="welcome"),
    path("attentionfailure", views.attentionfailure, name="attentionfailure"),
    path("consentform", views.consentform, name="consentform"),
    path("questionnaires", views.questionnaires, name="questionnaires"),
    path("instructions", views.instructions, name="instructions"),
    path("onepageexamplegentask", views.onePageExampleGenTask, name="onepageexamplegentask"),
    path('onepageexamplegenupdate', views.onePageExampleGenUpdate, name='onepageexamplegenupdate'),
    path("onepagecontextgentask", views.onePageContextGenTask, name="onepagecontextgentask"),
    path('onepagecontextgenupdate', views.onePageContextGenUpdate, name='onepagecontextgenupdate'),
    path("onepagediagnostic", views.onePageDiagnostic, name="onepagediagnostic"),
    path('onepagediagnosticupdate', views.onePageDiagnosticUpdate, name='onepagediagnosticupdate'),
    path("tutorial", views.tutorial, name="tutorial"),
    path("feedback", views.feedback, name="feedback"),
    path("goodbye", views.goodbye, name="goodbye"),
    path('token', views.token, name='token'),
    path('fishy', views.fishy, name='fishy'),
    path('alreadycompleted', views.alreadyCompleted, name='alreadycompleted')
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
