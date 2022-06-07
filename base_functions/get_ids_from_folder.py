import os
import random

"""
get_ids_from_folder(folders, max_images=None, seed=None, verbose=False):
This function checks all images (=tif-files) in the specified folders and returns the names of these images (=ids).
If multiple folders are entered, only the images are returned that exist in both folders (and not only in one for
example). If "max_images" is specified only a random subset of the images will be returned.
INPUT:
    folders (List): A list containing the paths to the folders in which the images are.
    max_images (int, None): How many images should be randomly selected and returned. If 'None' all images are returned.
    seed (String, None): A seed to have always the the same random images.
    catch (Boolean, True): If false and no common images can be found in the folders, the operation will exit python
    verbose (Boolean, False): If true, the status of the operations are printed
OUTPUT:
    result (list): A list of strings (ids, e.g. CAXXX)
"""


def get_ids_from_folder(folders, max_images=None, seed=None, catch=True, verbose=False):

    # if folders is just a path to a folder, put in a list
    if isinstance(folders, str):
        folders = [folders]

    assert len(folders) >= 1, "there must be at least one folder in the list"

    # all paths must be valid folders
    for elem in folders:
        assert os.path.isdir(elem), "'{}' is not a valid path".format(elem)

    # list of lists that has ids per folder
    all_files = []

    # iterate all folders
    for path in folders:

        if verbose:
            print("check files at {}".format(path))

        # save all ids
        files_list = []

        # iterate all files in folder
        for file in os.listdir(path):

            # only select tif files
            if file.endswith(".tif"):

                # remove the .tif
                files_list.append(file.split(".")[0])

        # save list to dict
        all_files.append(files_list)

    # get ids that are in all folders
    result = set(all_files[0])
    for s in all_files[1:]:
        result.intersection_update(s)

    # convert set to list
    result = list(result)

    # sort list so that seed is working
    result.sort()

    # get random selection of images
    if max_images is not None:
        if seed is not None:
            random.seed(seed)
        result = random.sample(result, max_images)

    # check if the results make sense
    if result is None or len(result) == 0:

        if catch is False:
            print("get_ids_from_folder: No common elements could be found in the folders")
            exit()
        else:
            result = []
    return result
