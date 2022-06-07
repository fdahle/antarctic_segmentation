import numpy as np
import rasterio
import warnings
import os

import download_images_from_usgs as difu

"""
load_image_from_file(image_id, image_type, image_path, catch, verbose):
This function loads an image from a specified path and returns it as a numpy array.
INPUT:
    image_id (String): The id of the image that should be loaded.
    image_type (String, "tif"): The type of image that should be loaded.
    image_path (String, None): The path where the image is located. If this is None the default aerial
        image path is used.
    download (Boolean, False): If true, the image will be downloaded if it is not available
    catch (Boolean, True): If true and somethings is going wrong, the operation will continue and not crash.
        In this case None is returned
    verbose (Boolean, False): If true, the status of the operations are printed
OUTPUT:
    img (np-array): The image loaded from the file
"""


def load_image_from_file(image_id, image_type="tif", image_path=None, download=False, catch=True, verbose=False):

    if image_path is None:
        image_path = "<Your default image path>"

    # ignore warnings of files not being geo-referenced
    warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

    # check if image path ends with '/' and add if not
    if image_path.endswith("/") is False:
        image_path = image_path + "/"

    # if the image type is already specified no need to add image type
    if len(image_id.split(".")) == 2:
        image_type = ""
    else:
        image_type = "." + image_type

    # create absolute path
    absolute_image_path = image_path + image_id + image_type

    if verbose:
        print("read {} from {}".format(image_id, absolute_image_path))

    # check if path exists (for the downloading)
    if download and os.path.exists(absolute_image_path) is False:
        _ = difu.download_images_from_usgs(image_id, image_path, verbose=verbose, catch=catch)

    try:
        ds = rasterio.open(absolute_image_path, 'r')

        # extract image from data
        img = ds.read()[0]

    except (Exception,) as e:
        if catch:
            return None
        else:
            raise e

    return img
