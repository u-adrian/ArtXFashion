from django import forms

class UploadFileForm(forms.Form):
    #title = forms.CharField(max_length=50)
    file = forms.ImageField()
    image = forms.ImageField()


class UploadImageForm(forms.Form):
    person_image_field = forms.ImageField()
    style_image_field = forms.ImageField()