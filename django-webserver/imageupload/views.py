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
#from somewhere import handle_uploaded_file




def handle_uploaded_file(upped_file, **kwargs):
    print('handling file function called')

    newtestfile = testfile(file_name=str(upped_file), file_image=upped_file)#, file_file=upped_file)

    newtestfile.save()
    #print(a)

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        print('post works')
        print(form)
        if form.is_valid():
            print('isvalid')
            handle_uploaded_file(request.FILES['file'])
            return HttpResponse("Success uploading")
            #return HttpResponseRedirect('/success/url/')
    else:
        print('else caught')
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})


def index(request):
  template = loader.get_template('myfirst.html')
  all_entries = testfile.objects.all()
  context = {
      'kek' : 'kekler',
      "list": all_entries
  }
  #for entity in all_entries:
  #    print(entity.file_name)
  return render(request, "myfirst.html", context)
  #return HttpResponse(template.render())

def indexdep(request):
    #UploadFileForm()
    return HttpResponse("Hello, world. You're at the polls index.")

######################
def landing(request):

    return render(request, 'index.html')



def handle_uploaded_images(person_image, style_image, x, y, **kwargs):
    print('handling file function called')

    #newtestfile = testfile(file_name=str(upped_file), file_image=upped_file)#, file_file=upped_file
    #newtestfile.save()

    newTransfer = Transfer(person_image=person_image, style_image=style_image, segment_start_x=x, segment_start_y=y)
    token = newTransfer.token
    newTransfer.save()
    ## save to queue

    return token


def upload_transfer(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        print('post works')
        print(form)
        if form.is_valid():
            print('isvalid')
            x, y = request.POST['x_coord'], request.POST['y_coord']
            token = handle_uploaded_images(request.FILES['person_image_field'], request.FILES['style_image_field'], x, y)
            #return HttpResponse(f"Success uploading: {uuid}")
            return HttpResponseRedirect(f'/status/{token}/')
    else:
        print('else caught')
        form = UploadImageForm()
    return render(request, 'upload_transfer.html', {'form': form})


def uuid_status(request, token):

    #transfer = Transfer.objects.get(pk=uuid)
    transfer = Transfer.objects.filter(pk=token)
    print(transfer.count())
    if transfer.exists():
        #template = loader.get_template('get_status.html')
        transfer = transfer[0]
        #all_entries = Transfer.objects.all()
        context = {
        'token' : token,
        "transfer": transfer
        }
        return render(request, "get_status.html", context)

    else:
        
        
        #template = loader.get_template('token_not_found.html', {'uuid' : uuid})

        return render(request, "token_not_found.html", {'token' : token, 'uploadurl' : reverse(upload_transfer)})
