import matplotlib.pyplot as plt
import numpy as np
from helper_functions import (
    load_image,
    im_convert,
    vgg_ready,
    get_features,
    gram_matrix,
    get_mbr_clothing_info,
)
import torch
import torch.optim as optim
from torchvision import models
import torch.nn


class NCTS:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_hyperparams()
        self._init_model()

    def _init_hyperparams(self):
        """Die Hyperparameter im Folgenden sind für den Prototypen fix (Für welche es Sinn macht einstellbar zu sein, wird noch erprobt)"""

        # Auswahl der Schichten und der Gewichte
        # Content Layer mal rausgenommen 'conv4_2':0.2,
        self.style_layers_and_style_weights = {
            "conv1_1": 1.0,
            "conv2_1": 0.75,
            "conv3_1": 0.2,
            "conv4_1": 0.2,
            "conv5_1": 0.2,
        }

        # Auwahl der Content Layer
        self.content_layer = "conv4_2"
        self.style_layers = list(self.style_layers_and_style_weights.keys())
        self.selected_layers = self.style_layers.__add__([self.content_layer])

        # Festlegung der alphas und des beta
        self.content_fashion_weight = 0.5  # alpha_fashion
        self.content_art_weight = 10  # alpha_art
        self.style_art_weight = 1e6  # beta

        # Festlegen der Optimierungsschritte und Learning Rate
        self.steps = 2000
        self.learning_rate = 0.03

    def _init_model(self):
        # laden der Feature Extraktion Schichten des VGG19, den Klassifizier Part benötigen wir nicht
        self.vgg = models.vgg19(pretrained=True).features

        # alle VGG Parameter werden eingefroren, da wir nicht trainieren wollen
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # Model auf vorhandenes Device umziehen
        self.vgg.to(self.device)

    def _optimize(
        self,
        transformed_clothing,
        fashion_image_features,
        art_image_features,
        style_grams,
    ):
        # transformed_clothing alle "show_every" Schritte anzeigen
        show_every = 100
        optimizer = optim.Adam([transformed_clothing], lr=self.learning_rate)

        for ii in range(1, self.steps + 1):
            transformed_fashion_image_features = get_features(
                transformed_clothing, self.vgg, self.selected_layers
            )
            # content_fashion loss
            content_fashion_loss = torch.mean(
                (
                    transformed_fashion_image_features[self.content_layer]
                    - fashion_image_features[self.content_layer]
                )
                ** 2
            )
            # content_art loss
            content_art_loss = torch.mean(
                (
                    transformed_fashion_image_features[self.content_layer]
                    - art_image_features[self.content_layer]
                )
                ** 2
            )
            # style loss
            # mit 0 initialisieren
            style_loss = 0
            # für jede Schicht den loss der jeweiligen Gram Matrix daraufrechnen
            for layer in self.style_layers_and_style_weights:
                transformed_fashion_image_feature = transformed_fashion_image_features[
                    layer
                ]
                transformed_fashion_image_gram = gram_matrix(
                    transformed_fashion_image_feature
                )
                _, d, h, w = transformed_clothing.shape

                style_gram = style_grams[layer]

                layer_style_loss = self.style_layers_and_style_weights[
                    layer
                ] * torch.mean((transformed_fashion_image_gram - style_gram) ** 2)

                style_loss += layer_style_loss / (d * h * w)

            total_loss = (
                self.content_art_weight * content_art_loss
                + self.style_art_weight * style_loss
                + self.content_fashion_weight * content_fashion_loss
            )
            # updaten des transformed_fashion_image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if ii % show_every == 0:
                print("Total loss: ", total_loss.item())
                plt.imshow(im_convert(transformed_clothing))
                plt.show()

        return transformed_clothing

    def perform_ncts(
        self,
        art_image_path="mock_data/images_art/Katze.png",
        fashion_image_path="mock_data/images_fashion/0744.jpg",
        fashion_mask_path="mock_data/images_tshirt_masks/0744.png",
    ):
        fashion_image_vgg = load_image(fashion_image_path, for_vgg=True)
        fashion_image_np = im_convert(fashion_image_vgg)
        fashion_mask = load_image(
            fashion_mask_path, shape=fashion_image_vgg.shape[2:], for_vgg=False
        )
        selected_clothing = fashion_image_np * fashion_mask

        # Creating a white image with dimension that can be calculated
        # from white_rectangle_size_pixel_position values -> called w_i_ncts
        mbr_shape, mbr_location = get_mbr_clothing_info(fashion_mask)
        w_i_ncts = np.zeros(mbr_shape, dtype=np.uint8)
        w_i_ncts[:, :] = [255, 255, 255]
        # Anpassen der w_incts dimension auf die Bild Maske und das Fashion Image
        cropped_fashion_mask = fashion_mask[mbr_location]
        cropped_selected_clothing = selected_clothing[mbr_location]

        art_image = load_image(art_image_path, shape=mbr_shape[:2], for_vgg=True).to(
            self.device
        )
        cropped_selected_clothing_vgg = (
            vgg_ready(cropped_selected_clothing).float().to(self.device)
        )

        # Feature Maps des Art und des Fashion Bildes speichern
        art_image_features = get_features(art_image, self.vgg, self.selected_layers)
        fashion_image_features = get_features(
            cropped_selected_clothing_vgg, self.vgg, self.selected_layers
        )

        # Gram Matrizen für jede Schicht bei Input des Art Bildes berechnen
        style_grams = {
            layer: gram_matrix(art_image_features[layer])
            for layer in art_image_features
        }

        # Erstellung unseres target transformed_fashion_image welches iterativ basierend auf dem fashion_image transformiert wird
        transformed_clothing = (
            vgg_ready(cropped_fashion_mask)
            .clone()
            .float()
            .to(self.device)
            .requires_grad_(True)
        )

        final_transformed_clothing = self._optimize(
            transformed_clothing,
            fashion_image_features,
            art_image_features,
            style_grams,
        )

        # 4. Creating a blank image with the same shape as the original fashion_image/fashion_mask (widht and heigts) and locating w_i_ncts with the white_rectangle_size_pixel_positions in the blank image
        # -> it is called b_i_ncts
        # (How the rest of the image which is not w_i_ncts looks like does not matter, because in the next step this b_i_ncts is multiplied with the fashion_mask anyway)
        b_i_ncts = np.zeros(fashion_image_np.shape)
        b_i_ncts[mbr_location] = (
            im_convert(final_transformed_clothing) * cropped_fashion_mask
        )

        resulting_fashion_image = fashion_image_np.copy()
        resulting_fashion_image[b_i_ncts > 0] = b_i_ncts[b_i_ncts > 0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(fashion_image_np)
        ax2.imshow(resulting_fashion_image)

        return resulting_fashion_image


if __name__ == "__main__":
    ai_designer = NCTS()
    image = ai_designer.perform_ncts()
