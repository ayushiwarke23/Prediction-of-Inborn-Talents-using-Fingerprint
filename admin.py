from django.contrib import admin
from django.utils.html import format_html
from .models import FingerprintImage, DmitModel

@admin.register(FingerprintImage)
class FingerprintImageAdmin(admin.ModelAdmin):
    list_display = (
        "finger_name",
        "uploaded_by",
        "image_tag",
        "processed_image_tag",
        "minutiae_endings",
        "minutiae_bifurcations",
        "ridge_density",
        "core_point",
        "delta_point",
        "ridge_count_core_delta",
        "created_at",
    )

    readonly_fields = (
        "image_tag",
        "processed_image_tag",
        "minutiae_endings",
        "minutiae_bifurcations",
        "ridge_density",
        "core_point",
        "delta_point",
        "ridge_count_core_delta",
        "created_at",
    )

    fields = (
        "dmit",
        "uploaded_by",
        "finger_name",
        "image",
        "processed_image",
        "image_tag",
        "processed_image_tag",
        "minutiae_endings",
        "minutiae_bifurcations",
        "ridge_density",
        "core_point",
        "delta_point",
        "ridge_count_core_delta",
        "created_at",
    )

    def image_tag(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" />', obj.image.url)
        return "-"
    image_tag.short_description = "Original Image"

    def processed_image_tag(self, obj):
        if obj.processed_image:
            return format_html('<img src="{}" width="100" />', obj.processed_image.url)
        return "-"
    processed_image_tag.short_description = "Preprocessed Image"

    def core_point(self, obj):
        return f"({obj.core_x:.1f}, {obj.core_y:.1f})"
    core_point.short_description = "Core (x,y)"

    def delta_point(self, obj):
        return f"({obj.delta_x:.1f}, {obj.delta_y:.1f})"
    delta_point.short_description = "Delta (x,y)"

    ordering = ("-created_at",)  # newest first
