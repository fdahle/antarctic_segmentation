import copy
import numpy as np
import cv2

from skimage.segmentation import expand_labels
from scipy import ndimage as ndi

import correct_image_orientation_sky as cios

"""
improve_segmented_image(input_segmented, input_probabilities, img_id, catch, verbose):
This function takes a segmented images and applies a set of different rules in order to improve the segmentations. 
Input_probabilities and img_id are not required, but can improve the correction even more.
INPUT:
    input_segmented (np-array): The unprocessed segmented image
    input_probabilities (np-array, None): The probabilities for each class
    img_id (String, None): The image id of the segmented image
    verbose (Boolean, False): If true, the status of the operations are printed
OUTPUT:
    segmented (np-array): The corrected segmented image)
"""


def improve_segmented_image(input_segmented, input_probabilities=None, img_id=None, verbose=False):

    # deep copy to not change the original
    segmented = copy.deepcopy(input_segmented)

    # correct the image orientation (so that the sky is always at the top)
    _, segmented, corrected = cios.correct_image_orientation_sky(None, segmented, img_id, return_bool=True)

    # deep copy of the probabilities
    if input_probabilities is not None:
        probabilities = copy.deepcopy(input_probabilities)
    else:
        # so that the ide is not complaining
        probabilities = None

    # resize images (so that the improving is faster)
    orig_shape = segmented.shape
    segmented = cv2.resize(segmented, (2000, 2000), interpolation=cv2.INTER_NEAREST)

    if input_probabilities is not None:
        probabilities_new = []
        for i, elem in enumerate(probabilities):
            probabilities_new.append(cv2.resize(elem, (2000, 2000), interpolation=cv2.INTER_NEAREST))
        probabilities = np.asarray(probabilities_new)

    # remove sky from images with V in filename
    def remove_sky(_segmented):

        _segmented[_segmented == 6] = 0
        _segmented = expand_labels(_segmented, distance=1000)

        return _segmented

    # if there are patches with the class unknown, these will be filled
    def fill_unknown(_segmented, probs):
        _segmented[_segmented == 7] = np.amax(probs, axis=0)[_segmented == 7]
        return _segmented

    # remove stuff in sky that should not be there
    def remove_clusters_in_sky(_segmented):
        min_perc = 20

        for j in range(_segmented.shape[0]):
            uniques, counts = np.unique(_segmented[j, :], return_counts=True)
            percentages = dict(zip(uniques, counts * 100 / len(segmented[j, :])))
            try:
                if percentages[6] > min_perc:
                    _segmented[0:j, :] = 6
            except (Exception,):
                pass

            if j % 100 == 0:
                min_perc = min_perc + 5

        binary = copy.deepcopy(segmented)

        # make binary for sky and non sky
        binary[binary != 6] = 0
        binary[binary == 6] = 1

        # cluster the image
        uv = np.unique(binary)
        s = ndi.generate_binary_structure(2, 2)
        cum_num = 0

        clustered = np.zeros_like(binary)
        for v in uv:
            labeled_array, num_features = ndi.label((binary == v).astype(int), structure=s)
            clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
            cum_num += num_features

        # count the pixels of each cluster and put it in a list
        unique, counts = np.unique(clustered, return_counts=True)
        clusters = np.column_stack([unique, counts])

        # sort clusters
        clusters_sorted = clusters[np.argsort(clusters[:, 1])]

        filled = np.ones_like(binary)

        # iterate all clusters and set clusters below the threshold to background
        for _elem in clusters_sorted:
            if _elem[1] < 50000:

                filled[clustered == _elem[0]] = 0
            else:
                break

        filled[filled != 0] = _segmented[filled != 0]

        # fill the background with the surrounding pixels
        filled = expand_labels(filled, distance=1000)

        return filled

    # enlarge sky to a complete row if bigger than threshold
    def enlarge_sky(_segmented):

        min_perc = 20

        for j in range(_segmented.shape[0]):
            uniques, counts = np.unique(_segmented[j, :], return_counts=True)
            percentages = dict(zip(uniques, counts * 100 / len(_segmented[j, :])))
            try:
                if percentages[6] > min_perc:
                    _segmented[0:j, :] = 6
            except (Exception,):
                pass

            if j % 100 == 0:
                min_perc = min_perc + 5

        return _segmented

    # remove very small patches
    def remove_small_patches(_segmented):

        temp = copy.deepcopy(_segmented)

        filled = np.zeros_like(temp)

        # cluster the image
        uv = np.unique(temp)
        s = ndi.generate_binary_structure(2, 2)
        cum_num = 0

        clustered = np.zeros_like(temp)
        for v in uv:
            labeled_array, num_features = ndi.label((temp == v).astype(int), structure=s)
            clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
            cum_num += num_features

        # count the pixels of each cluster and put it in a list
        unique, counts = np.unique(clustered, return_counts=True)
        clusters = np.column_stack([unique, counts])

        # sort clusters
        clusters_sorted = np.flip(clusters[np.argsort(clusters[:, 1])], axis=0)

        # iterate all clusters and set clusters below the threshold to background
        for _elem in clusters_sorted:
            if _elem[1] > 50:

                # get the class
                class_idx = np.where(clustered == _elem[0])
                class_idx = (class_idx[0][0], class_idx[1][0])
                elem_class = temp[class_idx]

                filled[clustered == _elem[0]] = elem_class
            else:
                break

        # fill the background with the surrounding pixels
        filled = expand_labels(filled, distance=1000)

        # return the new image
        return filled

    def check_logical_patches(_segmented, probs):

        temp = copy.deepcopy(_segmented)

        # cluster the image
        uv = np.unique(temp)
        s = ndi.generate_binary_structure(2, 2)
        cum_num = 0

        clustered = np.zeros_like(temp)
        for v in uv:
            labeled_array, num_features = ndi.label((temp == v).astype(int), structure=s)
            clustered += np.where(labeled_array > 0, labeled_array + cum_num, 0).astype(clustered.dtype)
            cum_num += num_features

        # count the pixels of each cluster and put it in a list
        unique, counts = np.unique(clustered, return_counts=True)
        clusters = np.column_stack([unique, counts])

        # get neighbouring segments
        # centers = np.array([np.mean(np.nonzero(clustered == j), axis=1) for j in unique])
        vs_right = np.vstack([clustered[:, :-1].ravel(), clustered[:, 1:].ravel()])
        vs_below = np.vstack([clustered[:-1, :].ravel(), clustered[1:, :].ravel()])
        b_neighbors = np.unique(np.hstack([vs_right, vs_below]), axis=1).T

        # delete neighbours with themselves
        b_neighbors = np.delete(b_neighbors, np.where(b_neighbors[:, 0] == b_neighbors[:, 1]), axis=0)

        for _elem in clusters:

            segment_id = _elem[0]
            orig_class = int(np.median(_segmented[clustered == segment_id]))

            # get neighbours of this segment
            neighbours = b_neighbors[b_neighbors[:, 0] == segment_id]

            neighbour_classes = []

            if len(neighbours) == 0:
                continue

            # iterate neighbours and get their classes
            for neighbour in neighbours[0]:
                neighbour_class = int(np.median(_segmented[clustered == neighbour]))

                neighbour_classes.append(neighbour_class)

            avg_pred_vals = {}

            # iterate all predictions
            for j, prediction in enumerate(probs):
                avg_pred = np.average(prediction[clustered == segment_id])
                avg_pred_vals[j] = avg_pred

            # water to rock/snow
            if orig_class == 4:

                if avg_pred_vals[4] - avg_pred_vals[3] < 0.1 and 4 not in neighbours:
                    _segmented[clustered == segment_id] = 2
                    if verbose:
                        print("Water to rocks")
                elif avg_pred_vals[4] - avg_pred_vals[2] < 0.001 and 4 not in neighbours:
                    _segmented[clustered == segment_id] = 2
                    if verbose:
                        print("Water to snow")

            # clouds to snow
            if orig_class == 5:
                if avg_pred_vals[5] - avg_pred_vals[2] < 0.01 and 5 not in neighbours:
                    _segmented[clustered == segment_id] = 2
                    if verbose:
                        print("Clouds to snow")

            if orig_class == 2:
                if avg_pred_vals[5] > 0.9 and 5 in neighbours:
                    _segmented[clustered == segment_id] = 5
                    if verbose:
                        print("Snow to clouds")

            # snow to ice
            # classes[(classes == 1) & (pred[0] > 0.7)] = 0
            # print("Snow to ice")

        return _segmented

    if (img_id is not None and "V" in img_id) or corrected is None:
        if verbose:
            print("remove sky segments")
        segmented = remove_sky(segmented)

    if input_probabilities is not None:
        if verbose:
            print("fill unknown segments")
        segmented = fill_unknown(segmented, probabilities)

    # if (img_id is not None and "V" not in img_id) or corrected is not None:
    if verbose:
        print("remove clusters in sky")
    segmented = remove_clusters_in_sky(segmented)

    if verbose:
        print("enlarge sky")
    segmented = enlarge_sky(segmented)

    if verbose:
        print("remove small patches")
    segmented = remove_small_patches(segmented)

    if input_probabilities is not None:
        if verbose:
            print("check logical patches")
        segmented = check_logical_patches(segmented, probabilities)

    if corrected:
        segmented = segmented[::-1, ::-1]

    # resize images back to original size
    segmented = cv2.resize(segmented, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

    return segmented
