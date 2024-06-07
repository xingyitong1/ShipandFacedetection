
from django.http import Http404

from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions
from rest_framework.views import APIView, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny, IsAdminUser,IsAuthenticatedOrReadOnly
from django.core.exceptions import PermissionDenied
from book.groups_permissions import check_user_group
from .permissions import IsOwnerOrReadOnly

@api_view(['GET'])
@permission_classes((permissions.IsAuthenticated,))
def apiOverview(request):
	check_user_group(request.user,'api')
	api_urls = {
		}

	return Response(api_urls)