import numpy as np

import get_ids_from_folder as giff
import load_image_from_file as liff
import cut_off_edge as coe
import segment_image as si

import display_multiple_images as dmi

"""
apply_model(select_type, path_model, model_id,  num_images):
This function applied a segmentation model on unseen images and displays the results. Always the latest model with this
provided id will be selected.

INPUT:
    select_type (String): Should random images be segmented ('random') or specific images ('ids')
    path_model (String): The path where the model is located
    model_id (String): The name of the *.pt or *.pth file
    path_folder_images (String): The path where the images are located
    num_images (int, 0): The number of random images that should be segmented. Only required if select_type='random'
    image_ids (List, []) THe ids of the images that should be segmented. Only required if select_type='ids'
OUTPUT:
    None
"""


select_type = "random"  # can be random or id
num_images = 10  # must be filled if select_type is 'random'
image_ids = []  # must be filled with ids if select_type is 'id'

path_folder_images = "<Enter your path to the folder with the images>"
path_folder_models = "<Enter your path to the folder where the models should be stored>"
model_id = "training_resized_all_4"


def apply_model(select_type, path_model, model_id, path_folder_images, num_images=0, image_ids=[]):

    if select_type == "random":
        image_ids = giff.get_ids_from_folder([path_folder_images], max_images=num_images)

    for image_id in image_ids:

        img_full = liff.load_image_from_file(image_id)

        img = coe.cut_off_edge(img_full, image_id)

        # normalize image
        #img = (img - np.min(img)) / (np.max(img) - np.min(img))

        pred, probabilities, highest_prob = si.segment_image(img, image_id, model_path=path_model, model_id=model_id)

        dmi.display_multiple_images([img, pred, highest_prob], subtitles=["raw", "pred", "highest_prob"])


if __name__ == "__main__":
    apply_model(select_type, path_folder_models, model_id, path_folder_images, num_images, image_ids)
