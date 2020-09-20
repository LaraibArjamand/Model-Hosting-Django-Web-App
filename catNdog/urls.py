from django.conf.urls import url
from .views import predict, home

urlpatterns = [
    url('hello', home, name="hello"),
    url('predict', predict, name='predict'),
]

