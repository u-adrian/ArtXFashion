from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import os
from helper_functions import *
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
import torch.nn

#Inputs! fashion_mask wird in der Praxis nicht benötigt, aktuell schon da wir die Ergebnisse der Segemnation noch nicht haben
art_image_pfad = "../images_art/Katze.png"
fashion_image_pfad = "../images_fashion/0744.jpg"
fashion_mask_pfad = "../images_tshirt_masks/0744.png" 

# Die Hyperparameter im Folgenden sind für den Prototypen fest (welche sinnvoll einstellabr sind wird noch erprobt)

# Auswahl der Schichten und der Gewichte
# Content Layer mal rausgenommen 'conv4_2':0.2,
style_layers_and_style_weights = {'conv1_1':1.,'conv2_1':0.75, 'conv3_1':0.2, 'conv4_1':0.2,'conv5_1':0.2}
# Auwahl der Content Layer
content_layer = 'conv4_2'
style_layers = list(style_layers_and_style_weights.keys())
selected_layers = style_layers.__add__([content_layer])

# Festlegung der alphas und des beta
content_fashion_weight = 0.5  # alpha_fashion
content_art_weight = 10 #alpha_art
style_art_weight = 1e6  # beta

#Festlegen der Optimierungsschritte und Learning Rate
steps = 5000
learning_rate = 0.03


# laden der Feature Extraktion Schichten des VGG19, den Klassifizier Part benötigen wir nicht
vgg = models.vgg19(pretrained=True).features

# alle VGG Parameter werden eingefroren, da wir nicht trainieren wollen
for param in vgg.parameters():
    param.requires_grad_(False)

# Model auf die GPU umziehen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg.to(device)

# laden des fashion 
fashion_image = load_image(fashion_image_pfad).to(device)
fashion_mask = load_image(fashion_image_pfad, shape=fashion_image.shape[-2:]).to(device)
selected_clothing = fashion_image * fashion_mask.clip(0, 1).to(device)



# 0. Get minimum bounding rectangle (mbr) of clothing
#Detecting rgb value of the background, mistakes could be avoided by comparing the rgb values of all corners
fashion_mask_numpy = im_convert(fashion_mask)

x_first_pixel_fashion = None
y_first_pixel_fashion = None


for row in range(fashion_mask_numpy.shape[0]):
  rgb_values_row = fashion_mask_numpy[row,:]
  if rgb_values_row.sum() >= 2.9 : # white pixel has a sum of RGB values of 3, 2.9 is chosen because of some tolerance issues
    y_first_pixel_fashion = row
    break
 
for row in range(fashion_mask_numpy.shape[0])[y_first_pixel_fashion:]:
  rgb_values_row = fashion_mask_numpy[row,:]
  if rgb_values_row.sum() <= 2.9: 
    y_last_pixel_fashion = row-1
    break

for col in range(fashion_mask_numpy.shape[1]):
  rgb_values_col = fashion_mask_numpy[:,col]
  if rgb_values_col.sum() >= 2.9 :
    x_first_pixel_fashion = col
    break

for col in range(fashion_mask_numpy.shape[1])[x_first_pixel_fashion:]:
  rgb_values_col = fashion_mask_numpy[:,col]
  if rgb_values_col.sum() <= 2.9:
    x_last_pixel_fashion = col-1
    break

# incorporate some confirmation method for real world practice eg. sum has to be over 3 iteration over the defined treshhold

#clockwise pixel position definition of the corners of the smallest rectangle which encompasses the fashion item (topleft, topright, bottomright, bottomleft )
white_rectangle_size_pixel_position = [[y_first_pixel_fashion,x_first_pixel_fashion],[y_first_pixel_fashion,x_last_pixel_fashion], [y_last_pixel_fashion,x_last_pixel_fashion],[y_last_pixel_fashion,x_first_pixel_fashion]]
white_rectangle_size_pixel_position

# 1. Creating a white image with dimension that can be calculated from white_rectangle_size_pixel_position values -> called w_i_ncts
mbr_shape = (y_last_pixel_fashion - y_first_pixel_fashion, x_last_pixel_fashion - x_first_pixel_fashion, 3)
w_i_ncts = np.zeros(mbr_shape, dtype=np.uint8)
w_i_ncts[:, :] = [255, 255, 255]
w_i_ncts.shape
in_transform(w_i_ncts).to(device)


# Anpassen der w_incts dimension auf die Bild Maske und das und das Fashion Image
mbr_location = np.s_[y_first_pixel_fashion:y_last_pixel_fashion,x_first_pixel_fashion:x_last_pixel_fashion,:]
cropped_fashion_mask = fashion_mask[mbr_location]
cropped_selected_clothing = selected_clothing[mbr_location].to(device)
print(fashion_mask.shape)


#Laden des Art Images
art_image = load_image(art_image_pfad, shape=mbr_shape[:2]).to(device)



# Feature Maps des Art und des Fashion Bildes speichern
art_image_features = get_features(art_image, vgg, selected_layers)
fashion_image_features = get_features(in_transform(cropped_selected_clothing).float().to(device), vgg, selected_layers)

# Gram Matrizen für jede Schicht bei Input des Art Bildes berechnen
style_grams = {layer: gram_matrix(art_image_features[layer]) for layer in art_image_features}

# Erstellung unseres target transformed_fashion_image welches iterativ basierend auf dem fashion_image transformiert wird
transformed_clothing = in_transform(cropped_fashion_mask).float().clone().requires_grad_(True).to(device)



#Durchführung des NCTS

# transformed_fashion_image alle "show_every" Schritte anzeigen
show_every = 100


optimizer = optim.Adam([transformed_clothing], lr=learning_rate)


for ii in range(1, steps+1):
    
    
    transformed_fashion_image_features = get_features(transformed_clothing, vgg, selected_layers)
    
    #content_fashion loss
    content_fashion_loss = torch.mean((transformed_fashion_image_features[content_layer] - fashion_image_features[content_layer])**2)


    # content_art loss
    content_art_loss = torch.mean((transformed_fashion_image_features[content_layer] - art_image_features[content_layer])**2)
    
    # style loss
    # mit 0 initialisieren
    style_loss = 0
    # für jede Schicht den loss der jeweiligen Gram Matrix daraufrechnen
    for layer in style_layers_and_style_weights:
  
        transformed_fashion_image_feature = transformed_fashion_image_features[layer]
        transformed_fashion_image_gram = gram_matrix(transformed_fashion_image_feature)
        _, d, h, w = transformed_clothing.shape
        
        style_gram = style_grams[layer]
        
        layer_style_loss = style_layers_and_style_weights[layer] * torch.mean(( transformed_fashion_image_gram - style_gram)**2)
        
        style_loss += layer_style_loss / (d * h * w)
        
    
    total_loss = content_art_weight * content_art_loss + style_art_weight * style_loss + content_fashion_weight * content_fashion_loss
    
    # updaten des transformed_fashion_image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    
    if  ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(im_convert(transformed_clothing))
        plt.show()

