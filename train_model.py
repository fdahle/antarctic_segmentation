import argparse
import copy
import os.path
import json
import warnings
import datetime
import random
import sys
import pathlib
import shutil

import sklearn.metrics as sm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_msssim import SSIM

# get current code-folder
workdir = str(pathlib.Path().resolve())[:-4]

# add base and display functions
sys.path.append(workdir + "/base_functions")
sys.path.append(workdir + "/display_functions")

import get_ids_from_folder as giff
import load_image_from_file as liff
import cut_off_edge as coe
import load_data_from_json as ldfj

import display_segmented as ds
import display_unet_subsets as dus

from classes.u_net import UNET
from classes.u_net_small import UNET_SMALL
from classes.image_data_set import ImageDataSet

# training params
params_training = {
    "model_name": "<Your model name>",
    "small_model": True,
    "max_epochs": 10000,  # how many epochs should maximal be trained
    "learning_rate": 0.001,  # how fast should the model learn
    "early_stopping": 0,  # when to stop when there's no improvement; 0 disables early stopping
    "loss_type": "cross_entropy",  # define which loss type should be used ("cross_entropy", "ssim")
    "input_layers": 1,  # just the grayscale, add more if anything else is added e.g. dem
    "output_layers": 6,  # equals the number of classes
    "kernel_size": 3,  # how big is the cnn kernel
    "batch_size": 16,  # how many images per batch
    "percentages": [80, 20, 0],  # how many images per training, validation and test (in percentage),
    "ignore_classes": [0, 7],  # which classes should be ignored (for weights, loss and cropping)
    "save_step": 1,  # after how many steps should the model be saved
    "seed": 123,  # random seed to enable reproducibility
    "device": 'gpu',  # can be ["auto", "cpu", "gpu"]
    "num_workers": 2,  # with how many parallel process should the data be loaded,
    "bool_continue_training": False,  # if this is true the model_name must already exist (at least the .json-file)
}

params_augmentation = {
    "methods": ["resized", "flipped", "rotation", "brightness", "noise"],  # "normalize"],
    "aug_size": 256,  # can be for "resized" or "cropped"
    "crop_type": "inverted",
    "crop_numbers": 16,  # how many random crops are extracted from an image at training?
}

# debug params
params_debugging = {
    "max_images": None,  # put None if you want to load all images
    "bool_save": True,
    "code_location": "local",  # can be server or local
    "fraction": 3,  # how much should be rounded when displaying or saving stats,
    "print_times": True,
    "additional_checks": False,
    "display_input_images": False,
    "display_subset_images_training": False,
    "display_subset_images_validation": False,
}

bool_verbose = True

# the images and the segmentes images should have the same names
path_folder_images = "<Enter your path to the folder with the images>"
path_folder_segmented = "<Enter your path to the folder with the segmented images>"
path_folder_models = "<Enter your path to the folder where the models should be stored>"

db_type = "FILES"

# it should be possible to call the function from arguments
parser = argparse.ArgumentParser()

parser.add_argument('--image_folder', metavar='image_folder', type=str, help='the path to the image folder')
parser.add_argument('--segmented_folder', metavar='segmented_folder', type=str, help='the path to the segmented folder')
parser.add_argument('--model_folder', metavar='model_folder', type=str, help='the path to the model folder')
parser.add_argument('--model_name', metavar='model_name', type=str, help='the name of the model')
parser.add_argument('--loss_type', metavar='loss_type', type=str, help='the loss used during training')
parser.add_argument('--batch_size', metavar='batch_size', type=int, help='the batch size used during training')
parser.add_argument('--bool_continue_training', metavar='bool_continue_training',
                    type=str, help='should the Training be continued')
parser.add_argument('--aug_method', metavar='aug_method', type=str, help="with which method will be augmented")
parser.add_argument('--bool_normalize', metavar="bool_normalize", type=str, help="should the data be normalized")

args = parser.parse_args()

if args.image_folder is not None:
    path_folder_images = args.image_folder
if args.segmented_folder is not None:
    path_folder_segmented = args.segmented_folder
if args.model_folder is not None:
    path_folder_models = args.model_folder

if args.model_name is not None:
    params_training["model_name"] = args.model_name
if args.loss_type is not None:
    params_training["loss_type"] = args.loss_type
if args.batch_size is not None:
    params_training["batch_size"] = args.batch_size
if args.bool_continue_training is not None:
    if args.bool_continue_training == "True":
        params_training["bool_continue_training"] = True
    elif args.bool_continue_training == "False":
        params_training["bool_continue_training"] = False
    else:
        print("The wrong parameter was used")
        exit()
if args.aug_method is not None:
    if args.aug_method == "resized":
        params_augmentation["methods"][0] = "resized"
        params_augmentation["aug_size"] = 1024
    elif args.aug_mehthod == "cropped":
        params_augmentation["methods"][0] = "cropped"
        params_augmentation["aug_size"] = 512
    else:
        print("The wrong parameter was used")
        exit()
if args.aug_method is not None:
    if args.bool_continue_training == "True":
        if params_augmentation["methods"][-1] != "normalize":
            params_augmentation["methods"].append("normalize")
    elif args.bool_continue_training == "False":
        if params_augmentation["methods"][-1] == "normalize":
            params_augmentation["methods"].pop()
    else:
        print("The wrong parameter was used")
        exit()

"""
train_model(input_images, input_labels, params_train, params_aug, params_debug, verbose):
This function takes the images and their labels and trains a model for semantic segmentation. The complete handling with
loading the images and augmentation is handled by the class 'image_data_set'. Look there for more information.
The model itself is in the class 'u_net'.
The general order of the this function is the following:
- put all images into the dataloader ('image_data_set')
- iterate epochs and for each epoch call 'train_one_epoch'
- in each epoch several batches are called
- Per epoch this is done first for training and then for validation
INPUT:
    input_images (dict): A dict with the raw images (without borders), the keys are the image ids (CAXXX)
    input_labels (dict): A dict with the segmented images (without borders), the keys are the image ids (CAXXX)
    params_train: Params for the training itself
    params_aug: Params if and how the images should be augmented
    params_debug: Params required during debugging
    verbose:
OUTPUT:
"""

print(f"Start training {params_training['model_name']}")


def train_model(input_images, input_labels, params_train,
                params_aug, params_debug, verbose=False):
    # track starting time for init
    start = datetime.datetime.now()

    if verbose:
        print("Start initializing model")

    # set device
    if params_train["device"] == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif params_train["device"] == "gpu":
        device = 'cuda'
    elif params_train["device"] == "cpu":
        device = 'cpu'
    else:  # last resort if everything is wrong
        print("A wrong device type is set, device is set to 'cpu'")
        device = 'cpu'

    debug_time = datetime.datetime.now()

    # initialize the model
    if verbose:
        print("Initialize model")
    if params_train["small_model"]:
        unet = UNET_SMALL(
            params_train["input_layers"],
            params_train["output_layers"],
            params_train["kernel_size"]
        )
    else:
        unet = UNET(
            params_train["input_layers"],
            params_train["output_layers"],
            params_train["kernel_size"]
        )
    unet = unet.to(device)

    if params_debug["print_times"]:
        difference = datetime.datetime.now() - debug_time
        print(f"Model initialized in {difference}")

    debug_time = datetime.datetime.now()

    # initialize the optimizer
    if verbose:
        print("Initialize optimizer")
    optimizer = torch.optim.Adam(unet.parameters(), lr=params_train["learning_rate"])

    if params_debug["print_times"]:
        difference = datetime.datetime.now() - debug_time
        print(f"Optimizer initialized in {difference}")

    debug_time = datetime.datetime.now()

    # initialize the datasets
    if verbose:
        print("Initialize dataset")
    dataset = ImageDataSet(input_images, input_labels, params_train, params_aug,
                           debug_print_times=params_debug["print_times"], verbose=verbose)

    if params_debug["print_times"]:
        difference = datetime.datetime.now() - debug_time
        print(f"Dataset initialized in {difference}")

    # a custom collate function to allow batches that contain different sizes
    def custom_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        return [data, target]

    # so that random is not so random
    def seed_worker(_):
        np.random.seed(params_train["seed"])
        random.seed(params_train["seed"])

    g = torch.Generator()
    g.manual_seed(params_train["seed"])
    torch.manual_seed(params_train["seed"])

    # parameters for the datasets
    params_dataset = {
        "batch_size": params_train["batch_size"],
        "num_workers": params_train["num_workers"],
        "collate_fn": custom_collate,
        "worker_init_fn": seed_worker,
        "generator": g
    }

    debug_time = datetime.datetime.now()

    # create the data loaders for train and validation
    if verbose:
        print("Initialize dataloader")
    train_dl = DataLoader(dataset, sampler=dataset.get_train_sampler(), **params_dataset)
    val_dl = DataLoader(dataset, sampler=dataset.get_valid_sampler(), **params_dataset)

    # create data loader for test
    if params_train["percentages"][2] > 0:
        test_dl = DataLoader(dataset, sampler=dataset.get_test_sampler(), **params_dataset)  # noqa

    if params_debug["print_times"]:
        difference = datetime.datetime.now() - debug_time
        print(f"Dataloaders initialized in {difference}")

    # if params_debug["additional_checks"]:
    #     train_ids = dataset.get_ids(category="train")
    #     val_ids = dataset.get_ids(category="validation")
    #     test_ids = dataset.get_ids(category="test")

    debug_time = datetime.datetime.now()

    # get the weights
    weights = dataset.get_weights(ignore=params_train["ignore_classes"])
    weights = torch.FloatTensor(weights)
    weights = weights.to(device)

    if params_debug["print_times"]:
        difference = datetime.datetime.now() - debug_time
        print(f"Weights calculated in {difference}")

    if params_debug["print_times"] is False:
        time_elapsed = datetime.datetime.now() - start
        print('Initialization complete in', str(time_elapsed))

    # track starting time for training
    start = datetime.datetime.now()

    if verbose:
        print("Start training on {}:".format(device))

    # define what should be saved
    saving_for = ["train", "val"]
    epoch_values = ["loss", "acc", "f1", "kappa"]
    best_values = ["best_loss_value", "best_loss_epoch",
                   "best_acc_value", "best_acc_epoch",
                   "best_f1_value", "best_f1_epoch",
                   "best_kappa_value", "best_kappa_epoch"]

    # create  starting epoch and the dicts for saving all parameters during training
    # if continue_training is true these dicts are not empty
    if params_train["bool_continue_training"] is False:

        # no time yet trained (this is required for the json
        time_from_previous_training = "0:0:0"
        training_number = 1

        # that will save later the params of the dataset (needed for different params per training)
        dataset_params = {}
        augmentation_params = {}

        epoch = 0

        # create the statistics dict
        statistics_dict = {}
        for elem1 in saving_for:
            for elem2 in epoch_values:
                statistics_dict[elem1 + "_" + elem2] = {}
            for elem2 in best_values:
                statistics_dict[elem1 + "_" + elem2] = 0
        statistics_dict["duration"] = {}

        # change loss to 100 (because the smallest loss is the best, not the highest)
        statistics_dict["train_best_loss_value"] = 100.0
        statistics_dict["val_best_loss_value"] = 100.0

    else:

        # specify json_path and load data
        json_path = path_folder_models + "/" + params_train["model_name"] + ".json"
        json_dict = ldfj.load_data_from_json(json_path)

        # get the time from the previous training
        time_from_previous_training = json_dict["training_time"]

        # load the statistic_dict
        statistics_dict = json_dict["statistics"]

        # get existing params from json
        dataset_params = json_dict["dataset_params"]
        augmentation_params = json_dict["params_augmentation"]

        # get the number of epochs
        epoch = len(statistics_dict["train_loss"])

        # get the number of trainings and increase
        training_number = len(json_dict["params_augmentation"]) + 1

        if epoch >= params_train["max_epochs"]:
            print(f"Please set a higher number of max_epochs (starting_epoch of {epoch} >= than "
                  f"max_epochs of {params_train['max_epochs']})")
            exit()

        # load model data (pth first, pt second)
        if os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".pth"):
            continue_model_path = path_folder_models + "/" + params_training["model_name"] + ".pth"

            date_time = datetime.datetime.now().strftime("%d_%d_%Y")
            copy_path = continue_model_path[:-4] + "_backup_" + date_time + ".pth"

            shutil.copyfile(continue_model_path, copy_path)

            checkpoint = torch.load(continue_model_path, map_location=device)

            # load model data to model
            unet.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        elif os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".pt"):
            continue_model_path = path_folder_models + "/" + params_training["model_name"] + ".pt"

            date_time = datetime.datetime.now().strftime("%d_%d_%Y")
            copy_path = continue_model_path[:-3] + "__backup_" + date_time + ".pt"

            shutil.copyfile(continue_model_path, copy_path)

            unet = torch.load(continue_model_path, map_location=device)

    def calc_loss(y_pred, y_true, input_params_train, input_weights):
        if input_params_train["loss_type"] == "cross_entropy":
            loss_fn = nn.CrossEntropyLoss(weight=input_weights, ignore_index=6)
            loss = loss_fn(y_pred, y_true)
        elif input_params_train["loss_type"] == "ssim":
            y_true_expanded = torch.unsqueeze(y_true, 1).double()
            y_argmax = torch.argmax(y_pred, 1)
            y_argmax_expanded = torch.unsqueeze(y_argmax, 1).double()

            ssim_module = SSIM(data_range=input_params_train["output_layers"], size_average=True, channel=1)
            loss = 1 - ssim_module(y_argmax_expanded, y_true_expanded)

        else:
            loss = None
            print("A wrong loss was set. Please select a different loss type")
            exit()

        return loss

    def calc_val(val_type, y_pred, y_true, input_weights):

        np_y_pred = y_pred.cpu().detach().numpy().flatten()
        np_y_true = y_true.cpu().detach().numpy().flatten()
        np_weights = input_weights.cpu().detach().numpy()
        weight_matrix = copy.deepcopy(np_y_pred).astype(float)

        # replace the idx in np_weights with the real weights
        for i, elem in enumerate(np_weights):
            weight_matrix[weight_matrix == i] = elem

        if val_type == "accuracy":

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                accuracy = sm.balanced_accuracy_score(np_y_true, np_y_pred, sample_weight=weight_matrix)

            return accuracy

        elif val_type == "f1":

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                f1 = sm.f1_score(np_y_true, np_y_pred, average="weighted", sample_weight=weight_matrix)

            return f1

        elif val_type == "kappa":

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                kappa = sm.cohen_kappa_score(np_y_true, np_y_pred, weights="linear", sample_weight=weight_matrix)

            return kappa

    def train_one_epoch(model, train_dataloader, val_dataloader, train_params, input_weights):

        epoch_start = datetime.datetime.now()

        # create one dict to save all avg values
        epoch_avg_statistics = {}

        ###
        # TRAINING
        ###

        # save loss and acc per batch to calculate the avg
        train_loss_batch = []
        train_acc_batch = []
        train_f1_batch = []
        train_kappa_batch = []

        batch_step = 0

        # iterate over all batches
        for train_x, train_y in train_dataloader:

            batch_start = datetime.datetime.now()

            batch_step += 1

            # stack must be different for cropped or resized
            if train_x[0].shape[0] == 1:
                # stack the data (from list to tensor)
                train_x = torch.stack(train_x)
                train_y = torch.stack(train_y)
            else:
                train_x = torch.cat(train_x, dim=0)
                train_y = torch.cat(train_y, dim=0)

                # add dummy dimension
                train_x = train_x[:, None, :, :]

            def train_subset(subset_x, subset_y):

                # zero the gradients
                optimizer.zero_grad()

                if params_debug["display_subset_images_training"]:
                    normalized = "normalize" in params_aug["methods"]
                    dus.display_unet_subsets(subset_x, subset_y, normalized, "Training")

                # data to gpu
                subset_x = subset_x.to(device)
                subset_y = subset_y.to(device)

                # predict
                subset_pred = model(subset_x)

                # get the max prediction for every cell
                subset_pred_max = torch.argmax(subset_pred, dim=1)

                # get loss, accuracy, f1 and kappa
                loss_for_subset = calc_loss(subset_pred, subset_y, train_params, input_weights)
                accuracy_for_subset = calc_val("accuracy", subset_pred_max, subset_y, input_weights)
                f1_for_subset = calc_val("f1", subset_pred_max, subset_y, input_weights)
                kappa_for_subset = calc_val("kappa", subset_pred_max, subset_y, input_weights)

                if train_params["loss_type"] == "ssim":
                    loss_for_subset.requires_grad = True

                # backpropagation
                loss_for_subset.backward()
                optimizer.step()

                # make the tensors to normal values
                loss_for_subset = loss_for_subset.cpu().detach().item()

                return loss_for_subset, accuracy_for_subset, f1_for_subset, kappa_for_subset

            # if now stuff is bigger than batch-size we need another loop
            if train_x.shape[0] > train_params["batch_size"]:

                subset_losses = []
                subset_accuracies = []
                subset_f1_scores = []
                subset_kappa_scores = []

                for i in range(0, train_x.shape[0], train_params["batch_size"]):

                    subset_min = i
                    subset_max = i + train_params["batch_size"]

                    if subset_max > train_x.shape[0]:
                        subset_max = train_x.shape[0]

                    train_x_subset = train_x[subset_min:subset_max, :, :, :]
                    train_y_subset = train_y[subset_min:subset_max, :, :]

                    subset_loss, subset_acc, subset_f1, subset_kappa = train_subset(train_x_subset, train_y_subset)

                    subset_losses.append(subset_loss)
                    subset_accuracies.append(subset_acc)
                    subset_f1_scores.append(subset_f1)
                    subset_kappa_scores.append(subset_kappa)

                    # print the statistics of one subset
                    if verbose:
                        print('    Current subset: {} - Loss: {} - Acc: {} - F1: {} - Kappa: {} - AllocMem (Mb): {}'.
                              format(
                                    batch_step,
                                    round(subset_loss, params_debug["fraction"]),
                                    round(subset_acc, params_debug["fraction"]),
                                    round(subset_f1, params_debug["fraction"]),
                                    round(subset_kappa, params_debug["fraction"]),
                                    round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                                    ))

                train_loss = np.mean(subset_losses)
                train_accuracy = np.mean(subset_accuracies)
                train_f1 = np.mean(subset_f1_scores)
                train_kappa = np.mean(subset_kappa_scores)
            else:
                train_loss, train_accuracy, train_f1, train_kappa = train_subset(train_x, train_y)

            # save loss and acc of one batch
            train_loss_batch.append(train_loss)
            train_acc_batch.append(train_accuracy)
            train_f1_batch.append(train_f1)
            train_kappa_batch.append(train_kappa)

            one_batch_duration = datetime.datetime.now() - batch_start

            # print the statistics of one batch
            if verbose:
                print('  Current batch: {} - Loss: {} - Acc: {} - F1: {} - Kappa: {} - AllocMem (Mb): {} ({})'.format(
                    batch_step,
                    round(train_loss, params_debug["fraction"]),
                    round(train_accuracy, params_debug["fraction"]),
                    round(train_f1, params_debug["fraction"]),
                    round(train_kappa, params_debug["fraction"]),
                    round(torch.cuda.memory_allocated() / 1024 / 1024, 4),
                    one_batch_duration
                ))

        # calculate the avg
        epoch_avg_statistics["avg_train_loss"] = sum(train_loss_batch) / len(train_loss_batch)
        epoch_avg_statistics["avg_train_acc"] = sum(train_acc_batch) / len(train_acc_batch)
        epoch_avg_statistics["avg_train_f1"] = sum(train_f1_batch) / len(train_f1_batch)
        epoch_avg_statistics["avg_train_kappa"] = sum(train_kappa_batch) / len(train_kappa_batch)

        one_epoch_duration = datetime.datetime.now() - epoch_start
        epoch_intermediate = datetime.datetime.now()

        # print the statistics of one complete epoch
        if verbose:
            print(' Current epoch: {} - avg. train Loss: {} - avg. train Acc: {} - '
                  'avg. train F1: {} - avg. train Kappa: {} ({})'.format(
                    epoch,
                    round(epoch_avg_statistics["avg_train_loss"], params_debug["fraction"]),
                    round(epoch_avg_statistics["avg_train_acc"], params_debug["fraction"]),
                    round(epoch_avg_statistics["avg_train_f1"], params_debug["fraction"]),
                    round(epoch_avg_statistics["avg_train_kappa"], params_debug["fraction"]),
                    one_epoch_duration
                  ))

        ###
        # VALIDATION
        ###

        # save loss and acc per batch to calculate the avg
        val_loss_batch = []
        val_acc_batch = []
        val_f1_batch = []
        val_kappa_batch = []

        batch_step = 0

        # iterate over all batches
        with torch.no_grad():  # required as otherwise the memory blows up
            for val_x, val_y in val_dataloader:

                batch_start = datetime.datetime.now()

                batch_step += 1

                # stack must be different for cropped or resized
                if val_x[0].shape[0] == 1:
                    # stack the data (from list to tensor)
                    val_x = torch.stack(val_x)
                    val_y = torch.stack(val_y)
                else:
                    val_x = torch.cat(val_x, dim=0)
                    val_y = torch.cat(val_y, dim=0)

                    # add dummy dimension
                    val_x = val_x[:, None, :, :]

                def val_subset(subset_x, subset_y):

                    if params_debug["display_subset_images_validation"]:
                        normalized = "normalize" in params_aug["methods"]
                        dus.display_unet_subsets(subset_x, subset_y, normalized, "Validation")

                    # data to gpu
                    subset_x = subset_x.to(device)
                    subset_y = subset_y.to(device)

                    # predict
                    subset_pred = model(subset_x)

                    # get the max prediction for every cell
                    subset_pred_max = torch.argmax(subset_pred, dim=1)

                    # get loss, accuracy
                    loss_for_subset = calc_loss(subset_pred, subset_y, train_params, input_weights)
                    accuracy_for_subset = calc_val("accuracy", subset_pred_max, subset_y, input_weights)
                    f1_for_subset = calc_val("f1", subset_pred_max, subset_y, input_weights)
                    kappa_for_subset = calc_val("kappa", subset_pred_max, subset_y, input_weights)

                    # make the tensors to normal values
                    loss_for_subset = loss_for_subset.cpu().detach().item()

                    return loss_for_subset, accuracy_for_subset, f1_for_subset, kappa_for_subset

                # if now stuff is bigger than batch-size we need another loop
                if val_x.shape[0] > train_params["batch_size"]:

                    subset_losses = []
                    subset_accuracies = []
                    subset_f1_scores = []
                    subset_kappa_scores = []

                    for i in range(0, val_x.shape[0], train_params["batch_size"]):
                        subset_min = i
                        subset_max = i + train_params["batch_size"]

                        if subset_max > val_x.shape[0]:
                            subset_max = val_x.shape[0]

                        val_x_subset = val_x[subset_min:subset_max, :, :, :]
                        val_y_subset = val_y[subset_min:subset_max, :, :]

                        subset_loss, subset_acc, subset_f1, subset_kappa = val_subset(val_x_subset, val_y_subset)

                        subset_losses.append(subset_loss)
                        subset_accuracies.append(subset_acc)
                        subset_f1_scores.append(subset_f1)
                        subset_kappa_scores.append(subset_kappa)

                        one_batch_duration = datetime.datetime.now() - batch_start

                        # print the statistics of one batch
                        if verbose:
                            print(
                                '    Current subset: {} - Loss: {} - Acc: {} - '
                                'F1: {} - Kappa: {} - AllocMem (Mb): {} ({})'.format(
                                    batch_step,
                                    round(subset_loss, params_debug["fraction"]),
                                    round(subset_acc, params_debug["fraction"]),
                                    round(subset_f1, params_debug["fraction"]),
                                    round(subset_kappa, params_debug["fraction"]),
                                    round(torch.cuda.memory_allocated() / 1024 / 1024, 4),
                                    one_batch_duration
                                ))

                    val_loss = np.mean(subset_losses)
                    val_accuracy = np.mean(subset_accuracies)
                    val_f1 = np.mean(subset_f1_scores)
                    val_kappa = np.mean(subset_kappa_scores)
                else:
                    val_loss, val_accuracy, val_f1, val_kappa = val_subset(val_x, val_y)

                # save loss and acc of one batch
                val_loss_batch.append(val_loss)
                val_acc_batch.append(val_accuracy)
                val_f1_batch.append(val_f1)
                val_kappa_batch.append(val_kappa)

                # print the statistics of one batch
                if verbose:
                    print('  Current batch: {} - Loss: {} - Acc: {} - F1: {} - Kappa: {} - AllocMem (Mb): {}'.format(
                        batch_step,
                        round(val_loss, params_debug["fraction"]),
                        round(val_accuracy, params_debug["fraction"]),
                        round(val_f1, params_debug["fraction"]),
                        round(val_kappa, params_debug["fraction"]),
                        round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                    ))

        # calculate the avg
        epoch_avg_statistics["avg_val_loss"] = sum(val_loss_batch) / len(val_loss_batch)
        epoch_avg_statistics["avg_val_acc"] = sum(val_acc_batch) / len(val_acc_batch)
        epoch_avg_statistics["avg_val_f1"] = sum(val_f1_batch) / len(val_f1_batch)
        epoch_avg_statistics["avg_val_kappa"] = sum(val_kappa_batch) / len(val_kappa_batch)

        one_epoch_duration = datetime.datetime.now() - epoch_intermediate

        # print the statistics of one complete epoch
        if verbose:
            print(' Current epoch: {} - avg. val Loss: {} - avg. val Acc: {} - '
                  'avg. val F1: {} - avg. val Kappa: {} - ({})'.format(
                    epoch,
                    round(epoch_avg_statistics["avg_val_loss"], params_debug["fraction"]),
                    round(epoch_avg_statistics["avg_val_acc"], params_debug["fraction"]),
                    round(epoch_avg_statistics["avg_val_f1"], params_debug["fraction"]),
                    round(epoch_avg_statistics["avg_val_kappa"], params_debug["fraction"]),
                    one_epoch_duration
                  ))

        # calculate complete epoch duration
        one_epoch_duration = str(datetime.datetime.now() - epoch_start)

        return epoch_avg_statistics, one_epoch_duration

    def update_statistics(input_epoch, stat_dict, epoch_stats, stats_epoch_duration):

        stat_dict["final_epoch"] = input_epoch

        # save the values of the epoch
        stat_dict["train_loss"][input_epoch] = epoch_stats["avg_train_loss"]
        stat_dict["train_acc"][input_epoch] = epoch_stats["avg_train_acc"]
        stat_dict["train_f1"][input_epoch] = epoch_stats["avg_train_f1"]
        stat_dict["train_kappa"][input_epoch] = epoch_stats["avg_train_kappa"]
        stat_dict["val_loss"][input_epoch] = epoch_stats["avg_val_loss"]
        stat_dict["val_acc"][input_epoch] = epoch_stats["avg_val_acc"]
        stat_dict["val_f1"][input_epoch] = epoch_stats["avg_val_f1"]
        stat_dict["val_kappa"][input_epoch] = epoch_stats["avg_val_kappa"]
        stat_dict["duration"][input_epoch] = stats_epoch_duration

        # for early stopping
        model_is_the_best = False

        # save the best value and epoch for train loss
        if epoch_stats["avg_train_loss"] < stat_dict["train_best_loss_value"]:
            stat_dict["train_best_loss_value"] = epoch_stats["avg_train_loss"]
            stat_dict["train_best_loss_epoch"] = input_epoch

        # save the best value and epoch for train acc
        if epoch_stats["avg_train_acc"] > stat_dict["train_best_acc_value"]:
            stat_dict["train_best_acc_value"] = epoch_stats["avg_train_acc"]
            stat_dict["train_best_acc_epoch"] = input_epoch

        # save the best value and epoch for train f1
        if epoch_stats["avg_train_f1"] > stat_dict["train_best_f1_value"]:
            stat_dict["train_best_f1_value"] = epoch_stats["avg_train_f1"]
            stat_dict["train_best_f1_epoch"] = input_epoch

        # save the best value and epoch for train kappa
        if epoch_stats["avg_train_kappa"] > stat_dict["train_best_kappa_value"]:
            stat_dict["train_best_kappa_value"] = epoch_stats["avg_train_kappa"]
            stat_dict["train_best_kappa_epoch"] = input_epoch

        # save the best value and epoch for val loss
        if epoch_stats["avg_val_loss"] < stat_dict["val_best_loss_value"]:
            stat_dict["val_best_loss_value"] = epoch_stats["avg_val_loss"]
            stat_dict["val_best_loss_epoch"] = input_epoch
            model_is_the_best = True

        # save the best value and epoch for val acc
        if epoch_stats["avg_val_acc"] < stat_dict["val_best_acc_value"]:
            stat_dict["val_best_acc_value"] = epoch_stats["avg_train_acc"]
            stat_dict["val_best_acc_epoch"] = input_epoch

        # save the best value and epoch for val f1
        if epoch_stats["avg_val_f1"] < stat_dict["val_best_f1_value"]:
            stat_dict["val_best_f1_value"] = epoch_stats["avg_train_f1"]
            stat_dict["val_best_f1_epoch"] = input_epoch

        # save the best value and epoch for val kappa
        if epoch_stats["avg_val_kappa"] < stat_dict["val_best_kappa_value"]:
            stat_dict["val_best_kappa_value"] = epoch_stats["avg_train_kappa"]
            stat_dict["val_best_kappa_epoch"] = input_epoch

        return statistics_dict, model_is_the_best

    def dump_statistics(dump_path, _training_number, _time_from_previous_training, _time_elapsed):

        # we want model name to be at the top of the json, therefore extract and delete in train-params
        model_name = params_train["model_name"]
        del params_train["model_name"]

        # save the current time and convert it to formats that will be used in the statistics
        now = datetime.datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M")

        training_iteration = "training_" + str(_training_number)

        # convert the time from the previous dicts to timedelta
        hours = int(_time_from_previous_training.split(":")[0])
        minutes = int(_time_from_previous_training.split(":")[1])
        seconds = float(_time_from_previous_training.split(":")[2])
        previous_time_delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # add old training time to new and format it
        train_time = previous_time_delta + _time_elapsed
        train_time_formatted = str(train_time)

        # add the parameters from the current training to the dict
        dataset_params[training_iteration] = dataset.get_params()
        augmentation_params[training_iteration] = params_aug

        # dump statistics
        json_dict_to_dump = {
            "model_name": model_name,
            "datetime": now_str,
            "training_time": train_time_formatted,
            "statistics": statistics_dict,
            "params_training": params_train,
            "params_augmentation": augmentation_params,
            "dataset_params": dataset_params
        }

        # dump to file
        with open(dump_path, 'w') as fp:
            json.dump(json_dict_to_dump, fp, indent=4)

        # add model name again (because we save the dict often intermediate
        params_train["model_name"] = model_name

        if verbose:
            print("Json successfully dumped to '{}'".format(dump_path))

    # need a counter for early_stopping
    early_stopping_counter = 0

    # training not in a for loop, so it can be ended with ctrl + c
    continue_loop = True
    while continue_loop:
        try:

            # increment number of epochs
            epoch += 1

            if verbose:
                print("-" * 10)
                print('Epoch {} of {}'.format(epoch, params_train["max_epochs"]))

            # the actual training
            epoch_statistics, epoch_duration = train_one_epoch(unet, train_dl, val_dl, params_train, weights)

            # evaluate the statistics
            statistics_dict, save_model_because_best = update_statistics(epoch, statistics_dict,
                                                                         epoch_statistics, epoch_duration)

            # init dict for best model (required so that pycharm does not complain)
            best_model_dict = None

            # check for early stopping
            if save_model_because_best is True and params_train["early_stopping"] > 0:
                best_model_dict = copy.deepcopy(unet.state_dict())
                early_stopping_counter = 0
            if save_model_because_best is False and params_train["early_stopping"] > 0:
                early_stopping_counter += 1

            # save intermediate model every xth step
            if epoch % params_train["save_step"] == 0 and params_debug["bool_save"] is True:

                # create path
                intermediate_model_path = path_folder_models + "/" + params_train["model_name"] + ".pth"

                # delete old file if existing
                if os.path.exists(intermediate_model_path):
                    os.remove(intermediate_model_path)

                if verbose:
                    print("Save intermediate model to '{}'".format(intermediate_model_path))

                # save all settings so that new training can start easily
                torch.save({
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, intermediate_model_path)

                # also saves the dict
                intermediate_dict_path = path_folder_models + "/" + params_train["model_name"] + ".json"
                if os.path.exists(intermediate_dict_path):
                    os.remove(intermediate_dict_path)

                # track intermediate time
                time_elapsed = datetime.datetime.now() - start

                dump_statistics(intermediate_dict_path, training_number, time_from_previous_training, time_elapsed)

                if verbose:
                    print("Save intermediate dict to '{}'".format(intermediate_dict_path))

            # stop if max number of epochs
            if epoch == params_train["max_epochs"]:
                continue_loop = False

            # stop if early stopping was reached
            if early_stopping_counter == params_train["early_stopping"] and params_train["early_stopping"] > 0:
                continue_loop = False
                unet.load_state_dict(best_model_dict)
                statistics_dict["final_epoch"] = statistics_dict["val_best_loss_epoch"]

        except KeyboardInterrupt:

            # don't continue training
            continue_loop = False

            # check if model should be saved (only required if save is true)
            if params_debug["bool_save"] is True:

                input_required = True

                while input_required:

                    input_save = input("Do you want to save the model ('y', 'n')")

                    if input_save == "y":
                        params_debug["bool_save"] = True
                        input_required = False

                    elif input_save == "n":
                        params_debug["bool_save"] = False
                        input_required = False

                    else:
                        print("Please enter 'y' or 'n'")

    # track ending time
    time_elapsed = datetime.datetime.now() - start

    if verbose:
        print('Training finished in', str(time_elapsed))

    # get the model_paths
    model_path = path_folder_models + "/" + params_train["model_name"] + ".pt"
    intermediate_model_path = path_folder_models + "/" + params_train["model_name"] + ".pth"
    json_path = path_folder_models + "/" + params_train["model_name"] + ".json"

    # only save if not in debug mode
    if params_debug["bool_save"] is False:
        if os.path.exists(intermediate_model_path):
            os.remove(intermediate_model_path)
        return

    # remove existing files
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(intermediate_model_path):
        os.remove(intermediate_model_path)
    if os.path.exists(json_path):
        os.remove(json_path)

    # save model
    torch.save(unet, model_path)

    if verbose:
        print("Model successfully saved to '{}'".format(model_path))

    # dump statistics to json
    dump_statistics(json_path, training_number, time_from_previous_training, time_elapsed)


if __name__ == "__main__":

    # if continue training the old model and json must exist
    if params_training["bool_continue_training"]:
        assert os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".pt") or \
               os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".pth"), \
               "No model file found to continue training"
        assert os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".json"), \
            "No statistics found to continue training"

    # get all ids from the folders
    folders = [path_folder_images, path_folder_segmented]
    list_ids = giff.get_ids_from_folder(folders, max_images=params_debugging["max_images"],
                                        seed=params_training["seed"])

    images = {}
    labels = {}
    for img_id in list_ids:
        image = liff.load_image_from_file(img_id, image_path=path_folder_images, verbose=bool_verbose)
        segmented = liff.load_image_from_file(img_id, image_path=path_folder_segmented, verbose=bool_verbose)

        if image is None or segmented is None:
            print("Loading for {} failed".format(img_id))
            continue

        # remove edge
        image = coe.cut_off_edge(image, img_id, db_type=db_type, verbose=bool_verbose, catch=False)
        segmented = coe.cut_off_edge(segmented, img_id, db_type=db_type, verbose=bool_verbose, catch=False)

        if image is None or segmented is None:
            print(f"Something went wrong with {img_id} and this image is skipped")
            continue

        # images must have the same shape, otherwise the model fails
        assert image.shape == segmented.shape, \
            f"Image shape {image.shape} is different from segmented shape {segmented.shape}"

        images[img_id] = image
        labels[img_id] = segmented

        if params_debugging["display_input_images"]:
            ds.display_segmented(image, segmented)

    train_model(images, labels, params_training, params_augmentation, params_debugging, verbose=bool_verbose)
