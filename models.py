from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_delete
from django.dispatch import receiver

class DmitModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)

class FingerprintImage(models.Model):
    dmit = models.ForeignKey(DmitModel, on_delete=models.CASCADE, related_name='fingerprints')
    image = models.ImageField(upload_to='fingerprints/')
    processed_image = models.ImageField(upload_to='preprocessed/', null=True, blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    finger_name = models.CharField(max_length=50, default="Unknown")

    # Extracted Features
    minutiae_endings = models.IntegerField(null=True, blank=True)
    minutiae_bifurcations = models.IntegerField(null=True, blank=True)
    ridge_density = models.FloatField(null=True, blank=True)
    core_x = models.FloatField(null=True, blank=True)
    core_y = models.FloatField(null=True, blank=True)
    delta_x = models.FloatField(null=True, blank=True)
    delta_y = models.FloatField(null=True, blank=True)
    ridge_count_core_delta = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

# Auto-delete files when record is deleted
@receiver(post_delete, sender=FingerprintImage)
def delete_files(sender, instance, **kwargs):
    if instance.image:
        instance.image.delete(save=False)
    if instance.processed_image:
        instance.processed_image.delete(save=False)
