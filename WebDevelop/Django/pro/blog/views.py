from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.

def index(request):
    return HttpResponse('<ul>'
                            '<li>我爱你</li>'
                            '<li>中国</li>'
                        '<ul/>')

def register(request):
    return HttpResponse('注册')