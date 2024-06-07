from django.shortcuts import render
from django.views.generic import TemplateView
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin
from util.useful import get_n_days_ago,create_clean_dir,change_col_format


TODAY=get_n_days_ago(0,"%Y%m%d")
PAGINATOR_NUMBER = 5
allowed_models = ['Category','Publisher','Book','Member','UserActivity']


# HomePage

class HomeView(LoginRequiredMixin,TemplateView):
    login_url = 'login'
    template_name = "index.html"
    context={}

  
    users = User.objects.all()
    for user in users:
        print(user.get_username(),user.is_superuser)

    def get(self,request, *args, **kwargs):

        # book_count = Book.objects.aggregate(Sum('quantity'))['quantity__sum']
        #
        # data_count = {"book":book_count,
        #             "member":Member.objects.all().count(),
        #             "category":Category.objects.all().count(),
        #             "publisher":Publisher.objects.all().count(),}
        #
        # user_activities= UserActivity.objects.order_by("-created_at")[:5]
        # user_avatar = { e.created_by:Profile.objects.get(user__username=e.created_by).profile_pic.url for e in user_activities}
        # short_inventory =Book.objects.order_by('quantity')[:5]
        #
        # current_week = date.today().isocalendar()[1]
        # new_members = Member.objects.order_by('-created_at')[:5]
        # new_members_thisweek = Member.objects.filter(created_at__week=current_week).count()
        # lent_books_thisweek = BorrowRecord.objects.filter(created_at__week=current_week).count()
        #
        # books_return_thisweek = BorrowRecord.objects.filter(end_day__week=current_week)
        # number_books_return_thisweek = books_return_thisweek.count()
        # new_closed_records = BorrowRecord.objects.filter(open_or_close=1).order_by('-closed_at')[:5]
        #
        # self.context['data_count']=data_count
        # self.context['recent_user_activities']=user_activities
        # self.context['user_avatar']=user_avatar
        # self.context['short_inventory']=short_inventory
        # self.context['new_members']=new_members
        # self.context['new_members_thisweek']=new_members_thisweek
        # self.context['lent_books_thisweek']=lent_books_thisweek
        # self.context['books_return_thisweek']=books_return_thisweek
        # self.context['number_books_return_thisweek']=number_books_return_thisweek
        # self.context['new_closed_records']=new_closed_records
 
        return render(request, self.template_name, self.context)
    
# Handle Errors

def page_not_found(request, exception):
    context = {}
    response = render(request, "errors/404.html", context=context)
    response.status_code = 404
    return response
    
def server_error(request, exception=None):
    context = {}
    response = render(request, "errors/500.html", context=context)
    response.status_code = 500
    return response
    
def permission_denied(request, exception=None):
    context = {}
    response = render(request, "errors/403.html", context=context)
    response.status_code = 403
    return response
    
def bad_request(request, exception=None):
    context = {}
    response = render(request, "errors/400.html", context=context)
    response.status_code = 400
    return response



