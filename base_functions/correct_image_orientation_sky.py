import copy
import numpy as np

"""
correct_image_orientation_sky(input_img, input_segmented, img_id, sky_id, return_bool, catch, verbose):
This function checks the location of the class 'sky' in the segmented images. If the sky is on the bottom (which should
not be the case in a real world), the image is turned 180 degrees. If no sky can found, the image remains untouched. If 
an image_id is provided, sometimes the correcting is faster (as vertical images with a 'V' in the id will not be 
touched). Note that the original input images are not changed (deep copy is applied before).
INPUT:
    input_img (np-array): The raw image that should be checked for right orientation and should possibly be corrected.
    input_segmented (np-array): The segmented version of the raw image.
    img_id (String, None): The image id of the input_img. Not required, but can help to speed up the process.
    sky_id (int, 6): The number of the class 'sky' in the segmented image. It is usually '6', but can be changed.
    return_bool (Boolean, False):  If true, the result of the correction is returned (True for rotating, False for not)
    catch (Boolean, True): If true and somethings is going wrong, the operation will continue and not crash
        not required here, as nothing can go wrong, just here for the sake of continuity
    verbose (Boolean, False): If true, the status of the operations are printed
OUTPUT:
    input_img (np-array): The raw image that has the right orientation
    input_segmented (np-array): The segmented version that that has the right orientation
    corrected (Boolean, optional): True if the image was rotated
"""


def correct_image_orientation_sky(input_img, input_segmented, img_id=None, sky_id=6, return_bool=False,
                                  catch=True, verbose=False):

    # if someone wants just to check the orientation of a segmented images just add None for input_img
    if input_img is None:
        input_img = np.empty([2, 2])

    # to satisfy the interpreter, but catch will not be used here
    if catch:
        pass

    # deep copy to not change the original
    img = copy.deepcopy(input_img)
    segmented = copy.deepcopy(input_segmented)

    if verbose:
        if img_id is None:
            print("Correct image orientation for image")
        else:
            print("Correct image orientation for {}".format(img_id))

    # no need to check the image if it is vertical
    if img_id is not None:
        if "V" in img_id:
            if verbose:
                print("Image has 'V' in id, so no orientation")

            if return_bool:
                return img, segmented, False
            else:
                return img, segmented

    # calculate percentages for top part of image
    uniques_top, counts_top = np.unique(segmented[0:200, :], return_counts=True)
    percentages_top = dict(zip(uniques_top, counts_top * 100 /
                               segmented[0:200, :].size))
    try:
        sky_top = percentages_top[sky_id]
        if sky_top < 1:
            sky_top = 0
    except (Exception,):
        sky_top = 0

    # calculate percentages for bottom part of image
    uniques_bottom, counts_bottom = np.unique(segmented[segmented.shape[0] - 200:, :], return_counts=True)
    percentages_bottom = dict(zip(uniques_bottom, counts_bottom * 100 /
                                  segmented[segmented.shape[0] - 200:, :].size))
    try:
        sky_bottom = percentages_bottom[sky_id]
        if sky_bottom < 0:
            sky_bottom = 0
    except (Exception,):
        sky_bottom = 0

    # image is right
    if sky_top > sky_bottom:
        if verbose:
            print("Image is orientated correctly")
        corrected = False

    # image is bottom
    elif sky_bottom > sky_top:
        if verbose:
            print("Image is orientated incorrectly")

        # switch images
        img = img[::-1, ::-1]
        segmented = segmented[::-1, ::-1]
        corrected = True

    # no sky (e.g. vertical)
    else:
        if verbose:
            print("Image has no orientation")
        corrected = None

    if return_bool:
        return img, segmented, corrected
    else:
        return img, segmented
