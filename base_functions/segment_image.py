import copy
import os
import glob
import warnings

import torch
import cv2
import numpy as np

import u_net as u_net


def segment_image(input_img, img_id=None,
                  model_id="LATEST_PTH", model_path=None,
                  resize_images=True, new_size=1200,
                  apply_threshold=False, min_threshold=0.9,
                  num_layers=1, num_output_layers=6,
                  verbose=False):

    # the segmentation has the following classes:
    # 1: ice, 2: snow, 3: rocks, 4: water, 5: clouds, 6:sky, 7: unknown

    if model_path is None:
        model_path = "<Path to the folder with your models">

    # deep copy to not change the original
    img = copy.deepcopy(input_img)

    # images must be in grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize image if wished (to increase the speed)
    if resize_images:
        orig_shape = img.shape
        img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
    else:
        # so that pycharm is not complaining
        orig_shape = None

    # init variable, so that pycharm is not complaining
    absolute_model_path = None

    # get absolute model path
    if model_id == "LATEST":
        # TODO ADD
        pass

    elif model_id == "LATEST_PT":
        list_of_files = glob.glob(model_path + "/*.pt")

        if len(list_of_files) == 0:
            print("segment_images (latest_pt): No model could be found at {}".format(model_path))
            exit()

        latest_model = max(list_of_files, key=os.path.getmtime)
        absolute_model_path = latest_model

    elif model_id == "LATEST_PTH":
        list_of_files = glob.glob(model_path + "/*.pth")

        if len(list_of_files) == 0:
            print("segment_images (latest_pth): No model could be found at {}".format(model_path))
            exit()

        latest_model = max(list_of_files, key=os.path.getmtime)
        absolute_model_path = latest_model

    else:
        absolute_model_path = model_path + "/" + model_id
        if os.path.exists(absolute_model_path + ".pt"):
            absolute_model_path = absolute_model_path + ".pt"
        elif os.path.exists(absolute_model_path + ".pth"):
            absolute_model_path = absolute_model_path + ".pth"
        else:
            print("Model {} is not existing at {}".format(model_id, absolute_model_path))
            exit()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        # set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # load the model
        if absolute_model_path.split(".")[-1] == "pt":
            model = torch.load(absolute_model_path)
            model.eval()
        elif absolute_model_path.split(".")[-1] == "pth":

            model = u_net.UNET(num_layers, num_output_layers)
            model.to(device)
            checkpoint = torch.load(absolute_model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        else:
            model = None
            print("This is not a valid file-type")
            exit()

    if verbose:
        if img_id is None:
            print("Segmentation for image is applied with model {}".format(absolute_model_path.split("/")[-1]))
        else:
            print("Segmentation for {} is applied with model {}".format(img_id, absolute_model_path.split("/")[-1]))

    # convert image so that it can be predicted
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)
    img = img.float()
    img = img.to(device)

    # the actual prediction 6 layers with prob for each layer
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        probabilities = model(img)
    probabilities = probabilities.cpu().detach().numpy()
    probabilities = probabilities[0]

    # 1 layer: per pixel the class with the highest prob
    pred = np.argmax(probabilities, axis=0)

    # 1 layer: per pixel the highest prob
    highest_prob = np.amax(probabilities, axis=0)

    if apply_threshold:
        pred[highest_prob < min_threshold] = 6

    # resize back to original
    if resize_images:
        _temp = np.empty((probabilities.shape[0], orig_shape[0], orig_shape[1]))
        for i, elem in enumerate(probabilities):
            _temp[i] = cv2.resize(elem, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        probabilities = _temp

        pred = cv2.resize(pred, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        highest_prob = cv2.resize(highest_prob, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    # pred to u int 8 to save some memory
    pred = pred.astype(np.uint8)

    pred = pred + 1

    return pred, probabilities, highest_prob


if __name__ == "__main__":
    import load_image_from_file as liff

    interesting_id = "CA181932V0005"

    example_img = liff.load_image_from_file(interesting_id)
    import cut_off_edge as coe

    example_img = coe.cut_off_edge(example_img)
    example_segmented, example_probabilities, _ = segment_image(example_img)

    import improve_segmented_image as isi

    segmented_improved = isi.improve_segmented_image(example_segmented, example_probabilities, verbose=True)

    import display_segmented as ds

    ds.display_segmented(example_img, segmented=example_segmented, segmented_improved=segmented_improved)
