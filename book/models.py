from django.db import models
from django.contrib.auth.models import AbstractUser
from django.contrib.auth.models import User
from django.urls import reverse

from PIL import Image

# UserProfile
class Profile(models.Model):
    user = models.OneToOneField(User,null=True,on_delete=models.CASCADE)
    bio = models.TextField()
    profile_pic = models.ImageField(upload_to="profile/%Y%m%d/", blank=True,null=True)
    phone_number = models.CharField(max_length=30,blank=True)
    email = models.EmailField(max_length=50,blank=True)

    def save(self, *args, **kwargs):
        # 调用原有的 save() 的功能
        profile = super(Profile, self).save(*args, **kwargs)

        # 固定宽度缩放图片大小
        if self.profile_pic and not kwargs.get('update_fields'):
            image = Image.open(self.profile_pic)
            (x, y) = image.size
            new_x = 400
            new_y = int(new_x * (y / x))
            resized_image = image.resize((new_x, new_y), Image.ANTIALIAS)
            resized_image.save(self.profile_pic.path)

        return profile

    def __str__(self):
        return str(self.user)

    def get_absolute_url(self): 
        return reverse('home')







