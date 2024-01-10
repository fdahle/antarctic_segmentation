import sys
import copy
import random
import math
import datetime
from typing import Dict, Any

import cv2
import numpy as np
import albumentations as album
import torch
from albumentations.pytorch.transforms import ToTensorV2
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class ImageDataSet(Dataset):

    def __init__(self, images, segmented, params_train, params_augmentation, debug_print_times=False, verbose=False):

        self.debug_print_times = debug_print_times
        self.verbose = verbose

        percentages = params_train["percentages"]

        assert images.keys() == segmented.keys()  # images and segmented must be identical
        assert len(percentages) == 3  # percentage for train, val and test
        assert sum(percentages) == 100  # percentages must add up to 100
        assert "resized" in params_augmentation["methods"] or "cropped" in params_augmentation[
            "methods"]  # images must be at least cropped or resized
        if "resized" in params_augmentation["methods"]:
            assert "cropped" not in params_augmentation["methods"]

        self.images = images  # the raw images
        self.segmented = segmented  # the segmented images

        debug_time = datetime.datetime.now()
        self.composition, self.composition_percentages = self.calc_composition()  # the composition per image
        if self.debug_print_times:
            difference = datetime.datetime.now() - debug_time
            print(f"  - Image composition calculated in {difference}")

        debug_time = datetime.datetime.now()
        self.labels = self.calc_labels()  # which classes are in which image
        if self.debug_print_times:
            difference = datetime.datetime.now() - debug_time
            print(f"  - Labels per image calculated in {difference}")

        self.percentages = percentages  # the percentages for train, val, test

        self.params_train = params_train  # the params for training
        self.params_augmentation = params_augmentation  # the params for augmentation

        # get the smallest image size of all images
        if "cropped" in params_augmentation["methods"]:
            min_width = sys.maxsize
            min_height = sys.maxsize

            for elem in self.images.values():
                if min_height > elem.shape[0]:
                    min_height = elem.shape[0]
                if min_width > elem.shape[1]:
                    min_width = elem.shape[1]

            self.cropped_min_width = min_width
            self.cropped_min_height = min_height

        # delete from images and segmented if not in composition (for images where the segmentation does not add up)
        for key in list(images.keys()):
            if key not in self.composition_percentages:
                print(f"WARNING: delete {key} from training")
                del self.images[key]
                del self.segmented[key]

        debug_time = datetime.datetime.now()

        # split images in training, val and test
        self.train_ids, self.val_ids, self.test_ids = self.split_in_sets()

        if self.debug_print_times:
            difference = datetime.datetime.now() - debug_time
            print(f"  - Dataset split in {difference}")

    def __len__(self):
        return len(self.images)

    # this function is called to get one item
    def __getitem__(self, input_id):

        img = self.images[input_id]
        segmented = self.segmented[input_id]

        # check if id in train, validation or test
        if input_id in self.train_ids:
            img, segmented = self.augment_data(img, segmented, "train")
        elif input_id in self.val_ids:
            img, segmented = self.augment_data(img, segmented, "val")
        elif input_id in self.test_ids:
            img, segmented = self.augment_data(img, segmented, "test")
        else:
            print("Image_dataset: This should not happen. Please check your code.")
            exit()

        # make the lowest value to 0 !!IMPORTANT!! MADE ME REINSTALL COMPLETE CUDA
        segmented = segmented - 1

        return img, segmented

    def get_ids(self, category):

        assert category in ["train", "validation", "test"], "This category is not existing"

        if category == "train":
            return self.train_ids
        elif category == "validation":
            return self.val_ids
        elif category == "test":
            return self.test_ids

    def get_train_sampler(self):
        train_sampler = OwnRandomSampler(self.train_ids)
        return train_sampler

    def get_valid_sampler(self):
        val_sampler = OwnRandomSampler(self.val_ids)
        return val_sampler

    def get_test_sampler(self):
        test_sampler = OwnRandomSampler(self.test_ids)
        return test_sampler

    def get_weights(self, ignore):

        if self.verbose:
            print("Calculate the weights")

        # init the pixel count
        total_counts = [0, 0, 0, 0, 0, 0, 0, 0]

        # count the number of pixels per class
        for key in self.train_ids:
            total_counts = [x + y for x, y in zip(total_counts, self.composition[key])]

        # delete the pixels that should be ignored
        for index in sorted(ignore, reverse=True):
            del [total_counts[index]]

        # convert to arr to make the following operations easier
        total_counts = np.asarray(total_counts)

        # replace 0 values with the lowest value (only required during debugging to prevent errors)
        min_val = np.amin(total_counts[np.nonzero(total_counts)])
        for i, elem in enumerate(total_counts):
            if elem == 0:
                total_counts[i] = min_val

        # get total count of pixels
        total_sum = sum(total_counts)

        # get weights
        weights = total_counts / total_sum

        # there should be no zero in weights
        for elem in weights:
            assert elem > 0

        # invert weights
        inverted = 1 / weights

        # normalize inverted weights
        normed = inverted / np.sum(inverted)

        # weights together should be 1
        assert round(sum(weights), 6) == 1

        return normed

    def get_params(self):

        params_dict = {"num_images": len(self.train_ids) + len(self.val_ids) + len(self.test_ids),
                       "num_train_images": len(self.train_ids),
                       "num_val_images": len(self.val_ids),
                       "num_test_images": len(self.test_ids),
                       "train_ids": self.train_ids,
                       "val_ids": self.val_ids,
                       "test_ids": self.test_ids}

        return params_dict

    # calculate the image composition
    def calc_composition(self):

        if self.verbose:
            print("  - Calculate image composition")

        comp_dict = {}
        comp_perc_dict = {}

        for key, val in self.segmented.items():

            # this list will contain the image composition
            # 0: undefined, 1: ice, 2:snow, 3:rocks, 4:water, 5:clouds, 6:sky, 7=unknown
            image_composition = [0, 0, 0, 0, 0, 0, 0, 0]
            image_percentages = [0, 0, 0, 0, 0, 0, 0, 0]

            # calculate percentages for the values
            labels, counts = np.unique(val, return_counts=True)
            total = np.sum(counts)
            percentages = np.round(counts / total, 3)

            # save the percentages
            for i, elem in enumerate(labels):
                image_composition[elem] = counts[i]
                image_percentages[elem] = percentages[i]

            # check for consistency, if positive result, add to dict
            if round(1 - sum(image_composition), 3) > 0.001:
                print("Invalid image composition for {} (sum is {})".format(key, sum(image_composition)))
            else:
                comp_dict[key] = image_composition
                comp_perc_dict[key] = image_percentages

        return comp_dict, comp_perc_dict

    # calculate which labels are in a segmented image and returns them as a list
    def calc_labels(self):

        if self.verbose:
            print("  - Calculate labels per image")

        label_dict = {}
        for key, val in self.segmented.items():
            labels = []
            for i in range(1, 7):  # ignore undefined and unknown
                if i in val:
                    labels.append(1)
                else:
                    labels.append(0)
            label_dict[key] = labels

        return label_dict

    def split_in_sets(self):

        if self.verbose:
            print("  - Split image in train, val, test")

        # get all ids from dict as list
        list_of_ids = list(self.images.keys())
        list_of_labels = np.asarray(list(self.labels.values()))

        # create the distribution of the first split as float values
        dist_1 = (self.percentages[0] / 100, (self.percentages[1] + self.percentages[2]) / 100)

        # first split in train and a temporary set (for test/val)
        # no seed required as shuffle=False
        stratifier = IterativeStratification(n_splits=2, sample_distribution_per_fold=dist_1)  #, shuffle=True,
                                             #random_state=self.params_train["seed"])

        # get the idx (=index number, not an id)
        temp_idx, train_idx = next(stratifier.split(X=list_of_ids, y=list_of_labels))

        # convert idx to ids (id = "CA-XX")
        train_ids = []
        for i in train_idx:
            train_ids.append(list_of_ids[i])

        temp_ids = []
        for i in temp_idx:
            temp_ids.append(list_of_ids[i])

        # no test set is required, so no need to split again
        if self.percentages[2] == 0:
            return train_ids, temp_ids, []

    def augment_data(self, img, segmented, set_type):

        class Brighto(album.ImageOnlyTransform):

            # abstract method that does nothing, but must be implemented
            def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
                pass

            def __init__(self, always_apply=False, p=0.5, val=50):
                super(Brighto, self).__init__(always_apply, p)
                self.val = val

            def apply(self, input_img, **params) -> np.ndarray:
                img_c = copy.deepcopy(input_img)

                if random.uniform(0, 1) > self.p:
                    brightness_change = random.randint(-self.val, self.val)

                    img_c = img_c + brightness_change
                    img_c[img_c < 0] = 0
                    img_c[img_c > 255] = 255

                return img_c

            def get_transform_init_args_names(self):
                tpl = ('val',)
                return tpl

        class Croppo(album.DualTransform):

            # abstract method that does nothing, but must be implemented
            def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
                pass

            # abstract method that does nothing, but must be implemented
            def apply_to_keypoint(self, keypoint, **params):
                pass

            # abstract method that does nothing, but must be implemented
            def apply_to_bbox(self, bbox, **params):
                pass

            def __init__(self, always_apply=True,
                         input_segmented=None, ignore=None,
                         crop_size=512, crop_type="random",
                         nr_of_crops=1):

                super(Croppo, self).__init__(always_apply)

                if ignore is None:
                    ignore = []
                assert crop_size % 2 == 0
                assert crop_type in ["weighted", "inverted", "random", "equally"]
                assert input_segmented.shape[0] > crop_size
                assert input_segmented.shape[1] > crop_size

                # divide crop field by 2 (because that's what we need to remove from the image)
                crop_half = int(crop_size / 2)

                # remove edges of the images
                seg_small = input_segmented[crop_half:input_segmented.shape[0] - crop_half,
                                            crop_half:input_segmented.shape[1] - crop_half]

                # get unique values of this subset
                unique, counts = np.unique(seg_small, return_counts=True)

                # remove the values that should be ignored
                idx_to_delete = []
                for elem in ignore:
                    idx = np.where(unique == elem)[0]
                    if len(idx) > 0:
                        idx_to_delete.append(idx)

                unique = np.delete(unique, idx_to_delete)
                counts = np.delete(counts, idx_to_delete)

                # count number of pixels in subset
                sum_pixels = seg_small.shape[0] * seg_small.shape[1]

                # get weights
                weights = counts / sum_pixels

                # save all crop boundaries as a list
                self.list_of_crop_boundaries = []

                for i in range(nr_of_crops):

                    # select a class from which the subset will be selected
                    if crop_type == "weighted":

                        # normalize weights
                        probs = weights / np.sum(weights)

                        # select based on weights
                        choice = unique[np.random.choice(len(unique), 1, p=probs)[0]]

                    elif crop_type == "inverted":

                        # invert weights
                        ln = 1 / weights

                        # normalize inverted weights
                        in_norm = ln / np.sum(ln)

                        # select based on inverted weights
                        choice = unique[np.random.choice(len(unique), 1, p=in_norm)[0]]

                    elif crop_type == "random":
                        choice = unique[np.random.choice(len(unique), 1)[0]]

                    elif crop_type == "equally":

                        # get number of classes
                        n_classes = len(unique)

                        # get probs
                        probs = [1 / n_classes] * n_classes

                        # select based on inverted probs
                        choice = unique[np.random.choice(len(unique), 1, p=probs)[0]]

                    # this should never happen
                    else:
                        choice = None

                    # get all indices of this class
                    indices = np.where(seg_small == choice)
                    indices = np.array(indices).T

                    # randomly select a pixel from this class
                    idx_row = np.random.choice(indices.shape[0], 1)[0]
                    coords = indices[idx_row, :]

                    # get the coords
                    y = coords[0]
                    x = coords[1]

                    # calculate extent of crop
                    min_y = y - int(crop_size / 2)
                    max_y = y + int(crop_size / 2)
                    min_x = x - int(crop_size / 2)
                    max_x = x + int(crop_size / 2)

                    # add the half crop size we removed before
                    min_y = min_y + int(crop_size / 2)
                    max_y = max_y + int(crop_size / 2)
                    min_x = min_x + int(crop_size / 2)
                    max_x = max_x + int(crop_size / 2)

                    self.list_of_crop_boundaries.append([min_y, max_y, min_x, max_x])

            def apply(self, input_img, **params) -> np.ndarray:

                all_crops = []
                for elem in self.list_of_crop_boundaries:
                    min_y = elem[0]
                    max_y = elem[1]
                    min_x = elem[2]
                    max_x = elem[3]
                    cropped = input_img[min_y:max_y, min_x:max_x]
                    all_crops.append(cropped)

                all_crops = np.asarray(all_crops)

                return all_crops

            def apply_to_mask(self, input_img, **params):

                all_crops = []
                for elem in self.list_of_crop_boundaries:
                    min_y = elem[0]
                    max_y = elem[1]
                    min_x = elem[2]
                    max_x = elem[3]
                    cropped = input_img[min_y:max_y, min_x:max_x]
                    all_crops.append(cropped)

                all_crops = np.asarray(all_crops)

                return all_crops

            def get_transform_init_args_names(self):
                return ()

        class Griddo(album.DualTransform):

            # abstract method that does nothing, but must be implemented
            def apply_to_bbox(self, bbox, **params):
                pass

            # abstract method that does nothing, but must be implemented
            def apply_to_keypoint(self, keypoint, **params):
                pass

            # abstract method that does nothing, but must be implemented
            def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
                pass

            def __init__(self, always_apply=True, crop_size=512):

                super(Griddo, self).__init__(always_apply)

                assert crop_size % 2 == 0

                self.crop_size = crop_size

            def grid_image(self, input_img):

                # get img size
                height, width = input_img.shape[0], input_img.shape[1]

                # crops must be smaller than img
                assert height >= self.crop_size
                assert width >= self.crop_size

                # check how often the crop fits into the image
                height_counter = height // self.crop_size
                width_counter = width // self.crop_size

                # check how much is left
                height_rest = height % self.crop_size
                width_rest = width % self.crop_size

                # get the extra between the crops
                extra_height = math.floor(height_rest / height_counter)
                extra_width = math.floor(width_rest / width_counter)

                # get the crops
                crops = []
                for y in range(height_counter):
                    for x in range(width_counter):
                        min_y = y * self.crop_size + y * extra_height
                        max_y = min_y + self.crop_size
                        min_x = x * self.crop_size + x * extra_width
                        max_x = min_x + self.crop_size

                        crop = input_img[min_y:max_y, min_x:max_x]
                        crops.append(crop)

                return np.asarray(crops)

            def apply(self, input_img, **params) -> np.ndarray:

                crops = self.grid_image(input_img)
                return crops

            def apply_to_mask(self, input_img, **params) -> np.ndarray:
                crops = self.grid_image(input_img)
                return crops

            # abstract method that does nothing, but must be implemented
            def get_transform_init_args_names(self):
                return ()

        class Normo(album.ImageOnlyTransform):

            # abstract method that does nothing, but must be implemented
            def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
                pass

            def __init__(self, always_apply=False, p=1.0):
                super(Normo, self).__init__(always_apply, p)

            def apply(self, input_img, **params) -> np.ndarray:
                img_c = copy.deepcopy(input_img)

                img_c = img_c / 128 - 1

                return img_c

            def get_transform_init_args_names(self):
                return ()

        aug_methods = self.params_augmentation["methods"]
        aug_size = self.params_augmentation["aug_size"]

        # save all augmentations
        augmentations1 = []

        """
        # for debugging
        augmentations1 = []
        augmentations2 = []
        augmentations3 = []
        augmentations4 = []
        augmentations5 = []
        augmentations6 = []
        augmentations7 = []
        augmentations8 = []
        """

        # init variable so that pycharm is not complaining
        gausso = None

        # these things are only for train
        if set_type == "train":
            if "noise" in aug_methods:
                gausso = album.GaussNoise(var_limit=(10, 50), p=0.5)
                augmentations1.append(gausso)

            if "flipping" in aug_methods:
                #augmentations1.append(album.VerticalFlip(p=0.5))
                augmentations1.append(album.HorizontalFlip(p=0.5))

            if "rotation" in aug_methods:
                augmentations1.append(album.RandomRotate90(p=0.5))

            if "brightness" in aug_methods:
                augmentations1.append(Brighto(val=25, p=0.5))

        if "normalize" in aug_methods:
            augmentations1.append(Normo(p=1.0))

        # create albums
        aug_album1 = album.Compose(augmentations1)

        # call gausso, otherwise it's not working (bug!?)
        if set_type == "train" and "noise" in aug_methods:
            _ = gausso.get_params_dependent_on_targets({"image": img})

        # apply augmentation
        augmented1 = aug_album1(image=img, mask=segmented)
        img_aug = augmented1["image"]
        segmented_aug = augmented1["mask"]

        # save all augmentations
        augmentations2 = []

        if "resized" in aug_methods:
            augmentations2.append(album.Resize(aug_size, aug_size, interpolation=cv2.INTER_NEAREST))

        elif "cropped" in aug_methods:
            if set_type == "train":
                augmentations2.append(Croppo(input_segmented=segmented_aug,
                                             crop_size=self.params_augmentation["aug_size"],
                                             crop_type=self.params_augmentation["crop_type"],
                                             ignore=self.params_train["ignore_classes"],
                                             nr_of_crops=self.params_augmentation["crop_numbers"]))
            elif set_type == "val":
                augmentations2.append(album.Resize(self.cropped_min_height, self.cropped_min_width,
                                                   interpolation=cv2.INTER_NEAREST))
                augmentations2.append(Griddo(crop_size=self.params_augmentation["aug_size"]))

        augmentations2.append(ToTensorV2())

        # create album
        aug_album2 = album.Compose(augmentations2)

        # apply second augmentation (for resize and cropped)
        augmented2 = aug_album2(image=img_aug, mask=segmented_aug)
        img_aug = augmented2["image"]
        segmented_aug = augmented2["mask"]

        # reverse the change of toTensor (only required if more than one image is there)
        if img_aug.shape[0] > 1:
            img_aug = np.transpose(img_aug, axes=[1, 2, 0])

        # convert to the required datatypes
        img_aug = img_aug.float()
        segmented_aug = segmented_aug.to(int)

        return img_aug, segmented_aug


class OwnRandomSampler(Sampler):

    def __init__(self, data_source, generator=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        # randomly shuffle the list
        shuffled = torch.randperm(n, generator=generator)

        # order the list based on the random shuffled
        shuffled_source = [_temp for _, _temp in sorted(zip(shuffled, self.data_source))] # noqa

        yield from shuffled_source
