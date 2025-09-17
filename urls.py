
from django.contrib import admin
from django.urls import path
from DMITapp.views import ulogin,signup,dashboard,ulogout,addfingerprint,delete_fingerprint,instruct
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path("",dashboard, name="dashboard"),
    path("ulogin",ulogin, name="ulogin"),
    path("instruct.html",instruct, name="instruct"),
    path("signup",signup,name="signup"),
    path("ulogout",ulogout, name="ulogout"),
    path("addfingerprint",addfingerprint, name="addfingerprint"),
    path("delete_fingerprint/<int:image_id>/", delete_fingerprint, name="delete_fingerprint"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
