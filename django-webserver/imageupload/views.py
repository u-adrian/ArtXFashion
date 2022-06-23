from django.shortcuts import render

from django.template import loader


from django.urls import reverse

# Create your views here.
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm, UploadImageForm
from .models import testfile, Transfer

# Imaginary function to handle an uploaded file.
# from somewhere import handle_uploaded_file
from .segmentation.cloth_segmentation import SegmentationModel

from django.core.files.images import get_image_dimensions
from django.core.files.images import ImageFile

from .styletransfer.NCTS import NCTS
from django.apps import AppConfig
from threading import Thread

import matplotlib.pyplot as plt
from os import listdir

from io import BytesIO
from PIL import Image
from django.core.files.images import ImageFile
import requests


### if more workers are included: refactor into diffrent module
## other solution for scalability with microservices 

class StyleTransferThread(Thread):
    def __init__(self, transfer):
        super(StyleTransferThread, self).__init__()
        self.transfer = transfer

    def run(self):

        transfer = self.transfer

        token = transfer.token

        x = transfer.segment_start_x
        y = transfer.segment_start_y
        model = SegmentationModel()
        pth = str(transfer.person_image)
        segmented_pil = model.do_image_segmentation(pth, x, y)
        path = f"images/segmentation/segmentation-{token}.png"
        saved = segmented_pil.save(path)
        transfer.person_image_segmentation = path
        transfer.save()


        result = self.do_style_transfer(
            transfer.style_image,
            transfer.person_image,
            transfer.person_image_segmentation,
        )
        path_style = f"images/style/style-{token}.png"
        saved = plt.imsave(path_style, result)
        transfer.style_transfered = path_style
        transfer.save()

    def do_style_transfer(self, a_img, f_img, f_mask):

        transfer_model = NCTS(self.transfer)
        return transfer_model.perform_ncts(
            art_image_path=a_img, fashion_image_path=f_img, fashion_mask_path=f_mask
        )



######################
def landing(request):

    return render(request, "index.html")


def convert(image):
    im = image
    im.convert("RGB")  # convert mode
    # im.thumbnail(size) # resize image
    thumb_io = BytesIO()  # create a BytesIO object
    im.save(thumb_io, "JPEG", quality=85)  # save image to BytesIO object
    thumbnail = File(thumb_io, name=image.name)  # create a django friendly File object








def handle_uploaded_images(person_image, style_image, x, y, **kwargs):

    # newtestfile = testfile(file_name=str(upped_file), file_image=upped_file)#, file_file=upped_file
    # newtestfile.save()

    width, height = get_image_dimensions(person_image)


    ##showing images at 300er resolution
    x = int(x)*width/300
    y = int(y)*width/300

    newTransfer = Transfer(
        person_image=person_image,
        style_image=style_image,
        segment_start_x=x,
        segment_start_y=y,
    )

    token = newTransfer.token
    newTransfer.save()
    ## save to queue

    # img_url = 'https://cdn.pixabay.com/photo/2021/08/25/20/42/field-6574455__340.jpg'

    # res = Image.open(requests.get(img_url, stream=True).raw)
    # filename = 'sample.jpeg'
    # img_object= ImageFile(BytesIO(segmented_pil.fp.getvalue()), name=filename)

    # django_image_field = img_object

    thread = StyleTransferThread(newTransfer)

    ### use .run() for debugging!!!
    thread.start()

    return token


def upload_transfer(request):

    art_image_list = listdir('images/preselection_art_images/')
    if request.method == "POST":
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            x, y = request.POST["x_coord"], request.POST["y_coord"]
            token = handle_uploaded_images(
                request.FILES["person_image_field"],
                request.FILES["style_image_field"],
                x,
                y,
            )
            # return HttpResponse(f"Success uploading: {uuid}")
            return HttpResponseRedirect(f"/status/{token}/")
    else:
        form = UploadImageForm()
    return render(request, "upload_transfer.html", {"form": form, "art_image_list" : art_image_list})


def uuid_status(request, token):

    # transfer = Transfer.objects.get(pk=uuid)
    transfer = Transfer.objects.filter(pk=token)
    print(transfer.count())
    if transfer.exists():
        # template = loader.get_template('get_status.html')
        transfer = transfer[0]
        # all_entries = Transfer.objects.all()
        context = {"token": token, "transfer": transfer}
        return render(request, "get_status.html", context)

    else:

        # template = loader.get_template('token_not_found.html', {'uuid' : uuid})

        return render(
            request,
            "token_not_found.html",
            {"token": token, "uploadurl": reverse(upload_transfer)},
        )
