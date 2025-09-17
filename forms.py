from django import forms
from .models import DmitModel, FingerprintImage

class DmitForm(forms.ModelForm):
    class Meta:
        model = DmitModel
        fields = "__all__"

class FingerprintImageForm(forms.ModelForm):
    class Meta:
        model = FingerprintImage
        fields = ['image']
