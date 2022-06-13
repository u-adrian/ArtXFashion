from PIL import Image
from torchvision import transforms
import numpy as np
import torch.nn

def load_image(img_path, max_size=400, shape=None):
    ''' Laden und transformieren von Bildern und Sicherstellung dass das Bild <= 400 pixels in der x-y dimension.'''
   
    image = Image.open(img_path).convert('RGB')
    
    # große Bilder beeinträchtigen die Ausführungszeit
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
    
    in_transform = transforms.Compose([
      transforms.Resize(size),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    # entfernen des transparenten alpha channels (:3) und hinzufügen der batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

 
def im_convert(tensor):
    """ Hilfsfunktion um ein Tensor Bild wieder darzustellen (un-normalizing, 
    konvertieren des Tensors in ein NumPy Bild), entnommen aus (2)"""
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    if image.shape[0] == 3: ## only do transopose wenn RGB
      image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def in_transform(image):
    '''Transformation eines Bildes in einen Tensor und Präperation für das VGG'''
    in_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    # entfernen des transparenten alpha channels (:3) und hinzufügen der batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image


# Methode um die Feature Maps einer spezifizierten Schicht auszugegeben
# basiert auf(2)
# ggf. ein paar Schichten rauslassen um den GPU Speicher nicht auszureizen


def get_features(image, model, selected_layers):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """             

    # Convolutional Layer und ihr Name in der VGG Definition
    dict_layers_name_to_num = {'conv1_1':'0',
                'conv1_2':'2',
                'conv2_1':'5', 
                'conv2_2':'7', 
                'conv3_1':'10', 
                'conv3_2':'12',
                'conv3_3':'14',
                'conv3_4':'16',
                'conv4_1':'19',
                'conv4_2':'21',
                'conv4_3':'23',
                'conv4_4':'25',
                'conv5_1':'28',
                'conv5_2':'30',
                'conv5_3':'32',
                'conv5_4':'34'}
    dict_layers_num_to_name = {v: k for k, v in dict_layers_name_to_num.items()}

    selected_layers_num = [dict_layers_name_to_num[layer] for layer in selected_layers]  

        
    features = {}
    x = image

    # model._modules ist ein dictionary in dem jede Schicht des Models gelistet ist
    for name, layer in model._modules.items():
        x = layer(x)
        if name in selected_layers_num:
            features[dict_layers_num_to_name[name]] = x
        
            
    return features

#Funktion zur Berechnung der Gram Matrix aus (2) entnommen
def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    # get the batch_size, depth, height, and width of the Tensor
    b, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(b * d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram 