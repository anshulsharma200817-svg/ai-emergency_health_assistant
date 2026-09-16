# pyrefly: ignore [missing-import]
from django.urls import path

from .views import RegisterView


urlpatterns = [
    path("register/", RegisterView.as_view(), name="register"),
]