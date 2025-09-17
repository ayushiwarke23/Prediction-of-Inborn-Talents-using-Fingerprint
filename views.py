from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.core.files import File
from random import choice
from pathlib import Path
from django.conf import settings
from django.core.files.base import ContentFile
from .models import DmitModel, FingerprintImage
from .preprocess import preprocess_and_save
from .feature_extraction import extract_features  # Your feature extraction module

@login_required
def addfingerprint(request):
    msg = ""
    error = ""
    fingers = {
        "Right_Thumb": "Right Thumb",
        "Right_Index": "Right Index",
        "Right_Middle": "Right Middle",
        "Right_Ring": "Right Ring",
        "Right_Little": "Right Little",
        "Left_Thumb": "Left Thumb",
        "Left_Index": "Left Index",
        "Left_Middle": "Left Middle",
        "Left_Ring": "Left Ring",
        "Left_Little": "Left Little",
    }

    if request.method == "POST":
        # Check if all files are uploaded
        missing_files = [name for name in fingers.keys() if name not in request.FILES]
        if missing_files:
            error = f"Please upload all fingerprint images."
        else:
            # All files uploaded, proceed
            dmit_instance, _ = DmitModel.objects.get_or_create(
                id=request.session.get("dmit_id"),
                defaults={}
            )
            request.session["dmit_id"] = dmit_instance.id

            for field_name, display_name in fingers.items():
                uploaded_file = request.FILES.get(field_name)

                # Delete old fingerprint
                FingerprintImage.objects.filter(
                    dmit=dmit_instance,
                    finger_name=display_name,
                    uploaded_by=request.user
                ).delete()

                # Save original
                fp = FingerprintImage.objects.create(
                    dmit=dmit_instance,
                    image=uploaded_file,
                    finger_name=display_name,
                    uploaded_by=request.user
                )

                # Preprocess image
                processed_file = preprocess_and_save(fp.image.path, f"{fp.id}.jpg")
                fp.processed_image.save(
                    f"{fp.id}.jpg",
                    ContentFile(processed_file.read()),
                    save=False
                )

                # Extract features and save in DB
                features = extract_features(fp.processed_image.path)
                for key, value in features.items():
                    setattr(fp, key, value)
                fp.save()

            msg = "All fingerprints uploaded, preprocessed, and features extracted successfully."

    # Fetch user images
    user_images = FingerprintImage.objects.filter(uploaded_by=request.user)

    return render(
        request,
        "addfingerprint.html",
        {
            "msg": msg,
            "error": error,
            "fingers": fingers,
            "user_images": user_images
        }
    )


from PIL import Image, ImageOps
from io import BytesIO

def preprocess_and_save(input_path, filename, target_size=(128,128)):
    """
    Preprocess a fingerprint and return an in-memory file (BytesIO).
    """
    img = Image.open(input_path)
    img = img.convert("L")            # grayscale
    img = ImageOps.equalize(img)      # improve contrast
    img = img.resize(target_size, Image.LANCZOS)

    temp_io = BytesIO()
    img.save(temp_io, format='JPEG')
    temp_io.seek(0)
    return temp_io


@login_required
def delete_fingerprint(request, image_id):
    image = get_object_or_404(FingerprintImage, id=image_id, uploaded_by=request.user)
    image.delete()
    return redirect("addfingerprint")



def dashboard(request):
	if request.user.is_authenticated:
		return render(request, "dashboard.html")
	else:
		return redirect("ulogin")

def signup(request):
	if request.user.is_authenticated:
		return redirect("dashboard")
	elif request.method == "POST":
		un = request.POST.get("un")
		try:
			usr = User.objects.get(username=un)
			msg = un + " already registered"
			return render(request,"signup.html",{"msg":msg})
		except User.DoesNotExist:
			txt = "0123456789abcdefghijklmnopABCDEFGHIJKLMNOP!@#$%^&*()QRSTUVWXYZqrstuvwxyz"
			pw = ""
			for i in range (1,9):
				pw = pw + choice(txt)	
			print(pw)
			usr = User.objects.create_user(username=un, password=pw)
			usr.save()
			subject = "Welcome to SmartDerm"
			text = "Your password is " + str(pw)
			from_email = "ayushiwarke69@apsit.edu.in"
			to_email=[str(un)]
			send_mail(subject, text, from_email,to_email)

			return redirect("ulogin")
	return render(request, "signup.html")

def ulogin(request):
	if request.user.is_authenticated:
		return redirect("dashboard")
	elif request.method == "POST":
		un = request.POST.get("un")
		pw = request.POST.get("pw")
		usr = authenticate(username=un,password=pw)
		if usr is None:
			msg = "Invalid username or password"
			return render(request,"ulogin.html",{"msg":msg})
		else:
			login(request,usr)
			return redirect("dashboard")
	else:
		return render(request,"ulogin.html")

def ulogout(request):
	logout(request)
	return redirect("ulogin")		

def instruct(request):
	return render(request,"instruct.html")