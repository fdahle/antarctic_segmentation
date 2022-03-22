import copy
import os.path
import json
import warnings
import datetime
import random
import sys
import pathlib

import sklearn.metrics as sm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


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
from classes.image_data_set import ImageDataSet

# training params
params_training = {
    "model_name": "train_resized",
    "max_epochs": 10000,  # how many epochs should maximal be trained
    "learning_rate": 0.1,  # how fast should the model learn
    "early_stopping": 0,  # when to stop when there's no improvement; 0 disables early stopping
    "loss_type": "crossentropy",  # define which loss type should be used
    "input_layers": 1,  # just the grayscale, add more if anything else is added e.g. dem
    "output_layers": 6,  # equals the number of classes
    "kernel_size": 3,  # how big is the cnn kernel
    "batch_size": 4,  # how many images per batch
    "percentages": [80, 20, 0],  # how many images per training, validation and test (in percentage),
    "ignore_classes": [0, 7],  # which classes should be ignored (for weights, loss and cropping)
    "save_step": 10,  # after how many steps should the model be saved
    "seed": 123,  # random seed to enable reproducibility
    "device": 'gpu',  # can be ["auto", "cpu", "gpu"]
    "num_workers": 2,  # with how many parallel process should the data be loaded,
    "bool_continue_training": True,  # if this is true the model_name must already exist (at least the .json-file)
}

params_augmentation = {
    "methods": ["resized", "flipped", "rotation", "brightness", "noise", "normalize"],
    "aug_size": 1024,  # can be for "resized" or "cropped"
    "crop_type": "inverted",
    "crop_numbers": 16,  # how many random crops are extracted from an image at training?
}

# debug params
params_debugging = {
    "max_images": None,  # put None if you want to load all images
    "bool_save": True,
    "fraction": 3,  # how much should be rounded when displaying or saving stats,
    "display_input_images": False,
    "display_subset_images": False
}

bool_verbose = True

path_folder_images = "../../data/aerial/TMA/downloaded"
path_folder_segmented = "../../data/aerial/TMA/segmented/supervised"
path_folder_models = "../../data/machine_learning/segmentation/UNET/models_new"

db_type = "FILES"

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

    # initialize the model
    unet = UNET(
        params_train["input_layers"],
        params_train["output_layers"],
        params_train["kernel_size"]
    )
    unet = unet.to(device)

    # initialize the optimizer
    if verbose:
        print("Initialize optimizer")
    optimizer = torch.optim.Adam(unet.parameters(), lr=params_train["learning_rate"])

    # initialize the datasets
    if verbose:
        print("Initialize dataset")
    dataset = ImageDataSet(input_images, input_labels, params_train, params_aug, verbose=verbose)

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

    # create the data loaders
    if verbose:
        print("Initialize dataloaders")
    train_dl = DataLoader(dataset, sampler=dataset.get_train_sampler(), **params_dataset)
    val_dl = DataLoader(dataset, sampler=dataset.get_valid_sampler(), **params_dataset)
    if params_train["percentages"][2] > 0:
        test_dl = DataLoader(dataset, sampler=dataset.get_test_sampler(), **params_dataset)

    # get the weights
    weights = dataset.get_weights(ignore=params_train["ignore_classes"])
    weights = torch.FloatTensor(weights)
    weights = weights.to(device)

    time_elapsed = datetime.datetime.now() - start
    print('Initialization complete in', str(time_elapsed))

    # track starting time for training
    start = datetime.datetime.now()

    if verbose:
        print("Start training on {}:".format(device))

    # define what should be saved
    saving_for = ["train", "val"]
    epoch_values = ["loss", "acc"]
    best_values = ["best_loss_value", "best_loss_epoch",
                   "best_acc_value", "best_acc_epoch"]

    # create  starting epoch and the dicts for saving all parameters during training
    # if continue_training is true these dicts are not empty
    if params_train["bool_continue_training"] is False:

        # no time yet trained (this is required for the json
        time_from_previous_training = "0:0:0"

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

        if epoch >= params_train["max_epochs"]:
            print(f"Please set a higher number of max_epochs (starting_epoch of {epoch} >= than "
                  f"max_epochs of {params_train['max_epochs']})")
            exit()

        # load model data (pth first, pt second)
        if os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".pth"):
            continue_model_path = path_folder_models + "/" + params_training["model_name"] + ".pth"
            checkpoint = torch.load(continue_model_path, map_location=device)

            # load model data to model
            unet.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        elif os.path.exists(path_folder_models + "/" + params_training["model_name"] + ".pt"):
            continue_model_path = path_folder_models + "/" + params_training["model_name"] + ".pt"
            unet = torch.load(continue_model_path, map_location=device)

    def calc_loss(y_pred, y_true, input_params_train, input_weights):
        if input_params_train["loss_type"] == "crossentropy":
            loss_fn = nn.CrossEntropyLoss(weight=input_weights, ignore_index=6)
        else:
            loss_fn = None
            print("A wrong loss was set. Please select a different loss type")
            exit()

        loss = loss_fn(y_pred, y_true)
        return loss

    def calc_accuracy(y_pred, y_true, input_weights):

        np_y_pred = y_pred.cpu().detach().numpy().flatten()
        np_y_true = y_true.cpu().detach().numpy().flatten()
        np_weights = input_weights.cpu().detach().numpy()
        weight_matrix = copy.deepcopy(np_y_pred).astype(float)

        # replace the idx in np_weights with the real weights
        for i, elem in enumerate(np_weights):
            weight_matrix[weight_matrix == i] = elem

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            accuracy = sm.balanced_accuracy_score(np_y_true, np_y_pred, sample_weight=weight_matrix)

        return accuracy

    def train_one_epoch(model, train_dataloader, val_dataloader, train_params, input_weights):

        # create one dict to save all avg values
        epoch_avg_statistics = {}

        ###
        # TRAINING
        ###

        # save loss and acc per batch to calculate the avg
        train_loss_batch = []
        train_acc_batch = []

        batch_step = 0

        # iterate over all batches
        for train_x, train_y in train_dataloader:

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

                if params_debug["display_subset_images"]:
                    normalized = "normalize" in params_aug["methods"]
                    dus.display_unet_subsets(subset_x, subset_y, normalized)

                # data to gpu
                subset_x = subset_x.to(device)
                subset_y = subset_y.to(device)

                # predict
                subset_pred = model(subset_x)

                # get the max prediction for every cell
                subset_pred_max = torch.argmax(subset_pred, dim=1)

                # get loss, accuracy
                loss_for_subset = calc_loss(subset_pred, subset_y, train_params, input_weights)
                accuracy_for_subset = calc_accuracy(subset_pred_max, subset_y, input_weights)

                # backpropagation
                loss_for_subset.backward()
                optimizer.step()

                # make the tensors to normal values
                loss_for_subset = loss_for_subset.cpu().detach().item()

                return loss_for_subset, accuracy_for_subset

            # if now stuff is bigger than batch-size we need another loop
            if train_x.shape[0] > train_params["batch_size"]:

                subset_losses = []
                subset_accuracies = []
                for i in range(0, train_x.shape[0], train_params["batch_size"]):
                    subset_min = i
                    subset_max = i + train_params["batch_size"]

                    if subset_max > train_x.shape[0]:
                        subset_max = train_x.shape[0]

                    train_x_subset = train_x[subset_min:subset_max, :, :, :]
                    train_y_subset = train_y[subset_min:subset_max, :, :]

                    subset_loss, subset_acc = train_subset(train_x_subset, train_y_subset)

                    subset_losses.append(subset_loss)
                    subset_accuracies.append(subset_acc)

                train_loss = np.mean(subset_losses)
                train_accuracy = np.mean(subset_accuracies)
            else:
                train_loss, train_accuracy = train_subset(train_x, train_y)

            # save loss and acc of one batch
            train_loss_batch.append(train_loss)
            train_acc_batch.append(train_accuracy)

            # print the statistics of one batch
            if verbose:
                print('  Current batch: {} - Loss: {} - Acc: {} -  AllocMem (Mb): {}'.format(
                    batch_step,
                    round(train_loss, params_debug["fraction"]),
                    round(train_accuracy, params_debug["fraction"]),
                    round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                ))

        # calculate the avg
        epoch_avg_statistics["avg_train_loss"] = sum(train_loss_batch) / len(train_loss_batch)
        epoch_avg_statistics["avg_train_acc"] = sum(train_acc_batch) / len(train_acc_batch)

        # print the statistics of one complete epoch
        if verbose:
            print(' Current epoch: {} - avg. train Loss: {} - avg. train Acc: {}'.format(
                epoch,
                round(epoch_avg_statistics["avg_train_loss"], params_debug["fraction"]),
                round(epoch_avg_statistics["avg_train_acc"], params_debug["fraction"]),
            ))

        ###
        # VALIDATION
        ###

        # save loss and acc per batch to calculate the avg
        val_loss_batch = []
        val_acc_batch = []

        batch_step = 0

        # iterate over all batches
        with torch.no_grad():  # required as otherwise the memory blows up
            for val_x, val_y in val_dataloader:

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

                    # zero the gradients
                    optimizer.zero_grad()

                    # data to gpu
                    subset_x = subset_x.to(device)
                    subset_y = subset_y.to(device)

                    # predict
                    subset_pred = model(subset_x)

                    # get the max prediction for every cell
                    subset_pred_max = torch.argmax(subset_pred, dim=1)

                    # get loss, accuracy
                    loss_for_subset = calc_loss(subset_pred, subset_y, train_params, input_weights)
                    accuracy_for_subset = calc_accuracy(subset_pred_max, subset_y, input_weights)

                    # make the tensors to normal values
                    loss_for_subset = loss_for_subset.cpu().detach().item()

                    return loss_for_subset, accuracy_for_subset

                # if now stuff is bigger than batch-size we need another loop
                if val_x.shape[0] > train_params["batch_size"]:

                    subset_losses = []
                    subset_accuracies = []
                    for i in range(0, val_x.shape[0], train_params["batch_size"]):
                        subset_min = i
                        subset_max = i + train_params["batch_size"]

                        if subset_max > val_x.shape[0]:
                            subset_max = val_x.shape[0]

                        val_x_subset = val_x[subset_min:subset_max, :, :, :]
                        val_y_subset = val_y[subset_min:subset_max, :, :]

                        subset_loss, subset_acc = val_subset(val_x_subset, val_y_subset)

                        subset_losses.append(subset_loss)
                        subset_accuracies.append(subset_acc)

                    val_loss = np.mean(subset_losses)
                    val_accuracy = np.mean(subset_accuracies)
                else:
                    val_loss, val_accuracy = val_subset(val_x, val_y)

                # save loss and acc of one batch
                val_loss_batch.append(val_loss)
                val_acc_batch.append(val_accuracy)

                # print the statistics of one batch
                if verbose:
                    print('  Current batch: {} - Loss: {} - Acc: {} -  AllocMem (Mb): {}'.format(
                        batch_step,
                        round(val_loss, params_debug["fraction"]),
                        round(val_accuracy, params_debug["fraction"]),
                        round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                    ))

        # calculate the avg
        epoch_avg_statistics["avg_val_loss"] = sum(val_loss_batch) / len(val_loss_batch)
        epoch_avg_statistics["avg_val_acc"] = sum(val_acc_batch) / len(val_acc_batch)

        # print the statistics of one complete epoch
        if verbose:
            print(' Current epoch: {} - avg. val Loss: {} - avg. val Acc: {}'.format(
                epoch,
                round(epoch_avg_statistics["avg_val_loss"], params_debug["fraction"]),
                round(epoch_avg_statistics["avg_val_acc"], params_debug["fraction"]),
            ))

        return epoch_avg_statistics

    def update_statistics(input_epoch, stat_dict, epoch_stats):

        stat_dict["final_epoch"] = input_epoch

        # save the values of the epoch
        stat_dict["train_loss"][input_epoch] = epoch_stats["avg_train_loss"]
        stat_dict["train_acc"][input_epoch] = epoch_stats["avg_train_acc"]
        stat_dict["val_loss"][input_epoch] = epoch_stats["avg_val_loss"]
        stat_dict["val_acc"][input_epoch] = epoch_stats["avg_val_acc"]

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

        # save the best value and epoch for val loss
        if epoch_stats["avg_val_loss"] < stat_dict["val_best_loss_value"]:
            stat_dict["val_best_loss_value"] = epoch_stats["avg_val_loss"]
            stat_dict["val_best_loss_epoch"] = input_epoch
            model_is_the_best = True

        # save the best value and epoch for val acc
        if epoch_stats["avg_val_acc"] < stat_dict["val_best_acc_value"]:
            stat_dict["val_best_acc_value"] = epoch_stats["avg_train_acc"]
            stat_dict["val_best_acc_epoch"] = input_epoch

        return statistics_dict, model_is_the_best

    def dump_statistics(dump_path):

        # we want model name to be at the top of the json, therefore extract and delete in train-params
        model_name = params_train["model_name"]
        del params_train["model_name"]

        # save the current time and convert it to formats that will be used in the statistics
        now = datetime.datetime.now()
        now_str = now.strftime("%d/%m/%Y %H:%M")
        now_fl_str = now.strftime("%d_%m_%Y_%H_%M")

        # convert the time from the previous dicts to timedelta
        hours = int(time_from_previous_training.split(":")[0])
        minutes = int(time_from_previous_training.split(":")[1])
        seconds = float(time_from_previous_training.split(":")[2])
        previous_time_delta = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)

        # add old training time to new and format it
        train_time = previous_time_delta + time_elapsed
        train_time_formatted = str(train_time)

        # add the parameters from the current training to the dict
        dataset_params[now_fl_str] = dataset.get_params()
        augmentation_params[now_fl_str] = params_aug

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
            epoch_statistics = train_one_epoch(unet, train_dl, val_dl, params_train, weights)

            # evaluate the statistics
            statistics_dict, save_model_because_best = update_statistics(epoch, statistics_dict, epoch_statistics)

            # init dict for best model (required so that pycharm does not complain)
            best_model_dict = None

            # check for early stopping
            if save_model_because_best is True and params_train["early_stopping"] > 0:
                best_model_dict = copy.deepcopy(unet.state_dict())
                early_stopping_counter = 0
            if save_model_because_best is False and params_train["early_stopping"] > 0:
                early_stopping_counter += 1

            # save intermediate model every xth step
            if epoch % params_train["save_step"] == 0:

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

                dump_statistics(intermediate_dict_path)

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
    dump_statistics(json_path)


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
        assert image.shape == segmented.shape

        images[img_id] = image
        labels[img_id] = segmented

        if params_debugging["display_input_images"]:
            ds.display_segmented(image, segmented)

    train_model(images, labels, params_training, params_augmentation, params_debugging, verbose=bool_verbose)
