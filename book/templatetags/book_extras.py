from django import template
from django.db.models.aggregates import Count
from django.utils import timezone
import math
import datetime
import requests
from django.template.loader import get_template

register = template.Library()

@register.inclusion_tag('book/inclusions/_pagination.html', takes_context=True)
def show_pagination(context):
    return {
        'page_objects':context['objects'],
        'search':context['search'],
        'orderby':context['orderby'],}

@register.inclusion_tag('book/inclusions/_messages.html',takes_context=True)
def show_messages(context):
        return {'messages':context['messages'],}
    
    
@register.simple_tag(takes_context=True)
def param_replace(context, **kwargs):
    """
    Based on
    https://stackoverflow.com/questions/22734695/next-and-before-links-for-a-django-paginated-query/22735278#22735278
    """
    d = context['request'].GET.copy()
    for k, v in kwargs.items():
        d[k] = v
    for k in [k for k, v in d.items() if not v]:
        del d[k]
    return d.urlencode()

@register.inclusion_tag('book/inclusions/_weather.html',takes_context=True)
# def show_weather(context):
#     url = 'http://api.openweathermap.org/data/2.5/weather?q=Chongqing,cn&units=metric&appid=2e37fd2364d867821f298280137eecc0'
#     r = requests.get(url).json()
#     chongqing_weather={}
#
#     if r['cod']==200:
#         chongqing_weather = {
#             'city': 'Chongqing',
#             'temperature': float("{0:.2f}".format(r['main']['temp'])),
#             'description': r['weather'][0]['description'],
#             'icon': r['weather'][0]['icon'],
#             'country': r['sys']['country']
#         }
#
#     return {'chongqing_weather':chongqing_weather}
def show_weather(context):
    url = 'http://api.openweathermap.org/data/2.5/weather?q=Chongqing,cn&units=metric&appid=2e37fd2364d867821f298280137eecc0'
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        weather_data = response.json()
    except requests.RequestException as e:
        # 处理请求异常
        print(f"请求失败: {e}")
        weather_data = None
    except ValueError as e:
        # 处理JSON解析错误
        print(f"JSON解析失败: {e}")
        weather_data = None

    chongqing_weather = {}

    if weather_data and weather_data.get('cod') == 200:
        chongqing_weather = {
            'city': 'Chongqing',
            'temperature': float("{0:.2f}".format(weather_data['main']['temp'])),
            'description': weather_data['weather'][0]['description'],
            'icon': weather_data['weather'][0]['icon'],
            'country': weather_data['sys']['country']
        }

    return {'chongqing_weather': chongqing_weather}