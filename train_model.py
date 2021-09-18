import json

import os
import time
import csv
import numpy as np
from copy import deepcopy
from datetime import datetime

from PIL import Image
import cv2

import torch
import torch.nn.functional as func
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors

from database_connection import Connector

with open('../../params1.json') as json_file:
    params1 = json.load(json_file)

verbose = True
seed = 123
fraction = 5

plot_overview = True
save_overview = True
plot_graphs = True
save_graphs = True
plot_test = True
save_test = True
save_stats = True
save_model = True

DEVICE = "auto"  # possible values are auto, cpu, gpu
BINARY = False

IMG_FOLDER = "../../../../data/aerial/TMA/downloaded"
MASK_FOLDER = "../../../../data/aerial/TMA/segmented/supervised"
SAVE_FOLDER = "../../../../data/models/segmentation/UNET"
OVERVIEW_FOLDER = "../../../../data/models/segmentation/UNET/overviews"
GRAPH_FOLDER = "../../../../data/models/segmentation/UNET/graphs"
TEST_FOLDER = "../../../../data/models/segmentation/UNET/tests"

augment = ["rotate", "mirror"]
IMG_SIZE = (800, 800)
edge = 0

train_perc = 80
val_perc = 10
test_perc = 10

MODEL_NAME = "model"
NR_EPOCHS = 250
LEARNING_RATE = 0.01
BATCH_SIZE = 5
EARLY_STOPPING = 50  # possible values are integer, if 0 there's no early stopping
LOSS_TYPE = "focal"  # possible values are focal, crossentropy

KERNEL_SIZE = 3
OUTPUT_LAYERS = 7


# own print function that only prints if verbose is True
def print_verbose(text, **kwarg):
    if verbose is False:
        return

    # distinguish between end or not
    if "end" in kwarg:
        print(text, end=kwarg["end"])
    else:
        print(text)


def get_file_list(path):
    init_files_list = []
    for filename in os.listdir(path):
        if filename.endswith(".tif"):
            init_files_list.append(filename)

    return init_files_list


# load the data and pre-process if necessary
def load_data(path, input_files_list, size=None, remove_edge=False, expand=False, random=False,
              input_augment=[], is_mask=False):
    conn = Connector(catch=False)

    def remove(input_img, input_filename):

        photo_id = input_filename[:-4]

        sql_string = "SELECT image_id, " + \
                     "fid_mark_1_x, " + \
                     "fid_mark_1_y, " + \
                     "fid_mark_2_x, " + \
                     "fid_mark_2_y, " + \
                     "fid_mark_3_x, " + \
                     "fid_mark_3_y, " + \
                     "fid_mark_4_x, " + \
                     "fid_mark_4_y " + \
                     "FROM images_properties " + \
                     "WHERE image_id='" + photo_id + "'"

        # get data from table
        table_data = conn.get_data(sql_string)

        subset_border = params1["unsupervised_subset_border"]

        if table_data is None:
            print("There is a problem with the table data. Please check your code")
            exit()

        # get left
        if table_data["fid_mark_1_x"].item() >= table_data["fid_mark_3_x"].item():
            left = table_data["fid_mark_1_x"].item()
        else:
            left = table_data["fid_mark_3_x"].item()

        # get top
        if table_data["fid_mark_2_y"].item() >= table_data["fid_mark_3_y"].item():
            top = table_data["fid_mark_2_y"].item()
        else:
            top = table_data["fid_mark_3_y"].item()

        # get right
        if table_data["fid_mark_2_x"].item() <= table_data["fid_mark_4_x"].item():
            right = table_data["fid_mark_2_x"].item()
        else:
            right = table_data["fid_mark_4_x"].item()

        # get bottom
        if table_data["fid_mark_1_y"].item() <= table_data["fid_mark_4_y"].item():
            bottom = table_data["fid_mark_1_y"].item()
        else:
            bottom = table_data["fid_mark_4_y"].item()

        left = int(left + subset_border)
        right = int(right - subset_border)
        top = int(top + subset_border)
        bottom = int(bottom - subset_border)

        input_img = input_img[top:bottom, left:right]

        return input_img

    # to store the image data
    data = []

    # iterate through the files
    for filename in os.listdir(path):
        if filename.endswith(".tif"):

            if filename not in input_files_list:
                continue

            # open and convert to array
            im = Image.open(path + "/" + filename)
            img = np.array(im)

            if remove_edge:
                img = remove(img, filename)

            if BINARY and is_mask:
                img[img <= 3] = 1
                img[img > 3] = 0

            # sometimes the images must be resized
            if size is not None:
                img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)

            data.append(img)

    # to store the augmentedData
    augmented_data = []

    if "rotate" in input_augment:
        print_verbose(" rotate data", end='\r')
        for elem in data:
            augmented_data.append(np.rot90(elem, 1))
            augmented_data.append(np.rot90(elem, 2))
            augmented_data.append(np.rot90(elem, 3))
        print_verbose(" rotate data - finished")

    if "mirror" in input_augment:
        print_verbose(" mirror data", end='\r')
        for elem in data:
            augmented_data.append(np.flip(elem, 0))
            augmented_data.append(np.flip(elem, 1))
        print_verbose(" mirror data - finished ")

    if "darker" in input_augment:
        print_verbose(" darken data", end='\r')
        for elem in data:
            random_brightness = np.random.normal(0.2, 0.8, 1)[0]
            augmented_data.append(elem * random_brightness)

    if len(augmented_data) > 0:
        data = [*data, *augmented_data]

    # add a band to the data (so that it is 3D): 1, W, H
    if expand:
        for i, elem in enumerate(data):
            data[i] = np.expand_dims(elem, axis=0)

    # convert data to np array
    data = np.asarray(data)

    # shuffle data
    if random:
        print_verbose(" randomize data", end='\r')
        np.random.seed(seed)
        np.random.shuffle(data)
        print_verbose(" randomize data - finished")

    print_verbose("There are in total {} images".format(data.shape[0]))

    return data


# split the data into test and val set
def split_data(data, input_percentages):
    print_verbose(" split data", end='\r')

    # set the random function seed
    torch.manual_seed(seed)

    # get sizes of test and val set
    train_size = int((data.shape[0] * input_percentages[0]) / 100)
    val_size = int((data.shape[0] * input_percentages[1]) / 100)
    test_size = int((data.shape[0] * input_percentages[2]) / 100)

    if val_size == 0:
        val_size = 1
        train_size = train_size - 1

    if test_size == 0:
        test_size = 1
        train_size = train_size - 1

    # check if numbers add up (and if not add to train size)
    left_over = data.shape[0] - train_size - val_size - test_size
    train_size = train_size + left_over

    # split data
    train_subset, val_subset, test_subset = random_split(data, [train_size, val_size, test_size])

    train_indices = train_subset.indices
    val_indices = val_subset.indices
    test_indices = test_subset.indices

    train_dataset = data[train_indices]
    val_dataset = data[val_indices]
    test_dataset = data[test_indices]

    print_verbose(" split data - finished (train-set: {}, validation-set: {}, test-set: {})".format(train_size,
                                                                                                    val_size,
                                                                                                    test_size))

    return train_dataset, val_dataset, test_dataset


# the model itself
class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, binary=False):
        super().__init__()

        self.binary = binary

        self.conv1 = self.contract_block(in_channels, 32, KERNEL_SIZE, 1)
        self.conv2 = self.contract_block(32, 64, KERNEL_SIZE, 1)
        self.conv3 = self.contract_block(64, 128, KERNEL_SIZE, 1)
        self.conv4 = self.contract_block(128, 256, KERNEL_SIZE, 1)

        self.upconv4 = self.expand_block(256, 128, KERNEL_SIZE, 1, output_padding=1)
        self.upconv3 = self.expand_block(128 * 2, 64, KERNEL_SIZE, 1, output_padding=1)
        self.upconv2 = self.expand_block(64 * 2, 32, KERNEL_SIZE, 1, output_padding=1)
        self.upconv1 = self.expand_block(32 * 2, out_channels, KERNEL_SIZE, 1, output_padding=1)

        self.sigmoid = nn.Sigmoid()

    def __call__(self, input_x):
        # downsampling part
        conv1 = self.conv1(input_x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        upconv4 = self.upconv4(conv4)
        upconv3 = self.upconv3(torch.cat([upconv4, conv3], 1))
        upconv2 = self.upconv2(torch.cat([upconv3, conv2], 1))
        upconv1 = self.upconv1(torch.cat([upconv2, conv1], 1))

        if self.binary:
            upconv1 = self.sigmoid(upconv1)

        return upconv1

    @staticmethod
    def contract_block(in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels,
                            kernel_size=kernel_size, stride=(1, 1), padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels,
                            kernel_size=kernel_size, stride=(1, 1), padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        return contract

    @staticmethod
    def expand_block(in_channels, out_channels, kernel_size, padding, output_padding=0):
        expand = nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                               torch.nn.BatchNorm2d(out_channels),
                               torch.nn.ReLU(),
                               torch.nn.Conv2d(
                                   out_channels, out_channels, kernel_size, stride=(1, 1), padding=padding),
                               torch.nn.BatchNorm2d(out_channels),
                               torch.nn.ReLU(),
                               torch.nn.ConvTranspose2d(
                                   out_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                   output_padding=output_padding)
                               )
        return expand


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = func.binary_cross_entropy(inputs.squeeze(), targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


class MultiFocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(MultiFocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input_data, target):
        ce_loss = func.cross_entropy(input_data, target, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


# function to calculate the accuracy
def acc_fn(predb, yb):
    equal = predb == yb

    true_vals = equal.sum()
    all_vals = equal.size
    accuracy = true_vals / all_vals

    return accuracy


# train the actual model
def train(model, input_train_dl, input_valid_dl, optimizer, input_loss_fn, input_acc_fn, epochs=100):
    print('-' * 10)
    print("Start training:")

    # track starting time
    start = time.time()

    # set model to device
    model = model.to(device)

    # save the values per epoch
    train_loss_epoch = []
    train_acc_epoch = []
    val_loss_epoch = []
    val_acc_epoch = []

    # to save the best values and their pos
    train_best_loss = 100.0
    # train_best_loss_epoch = 0
    train_best_acc = 0.0
    # train_best_acc_epoch = 0
    val_best_loss = 100.0
    val_best_loss_epoch = 0
    val_best_acc = 0.0
    # val_best_acc_epoch = 0

    best_model_dict = None

    for epoch in range(epochs):

        print('Epoch {}/{}'.format(epoch, epochs - 1))

        train_loss_batch = []
        train_acc_batch = []

        step = 0
        # iterate over data per batch
        for input_train_x, input_train_y in input_train_dl:

            step += 1

            # data to gpu
            input_train_x = input_train_x.to(device)
            input_train_y = input_train_y.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # predict
            train_preds = model(input_train_x)

            # get loss
            train_loss = input_loss_fn(train_preds, input_train_y)

            # backpropagation
            train_loss.backward()
            optimizer.step()

            if BINARY:
                classes = (train_preds > 0.5).cpu().detach().numpy()
            else:
                classes = np.argmax(train_preds.cpu().detach().numpy(), axis=1)

            # get accuracy
            train_acc = input_acc_fn(classes, input_train_y.cpu().detach().numpy())

            # save the values
            train_loss_batch.append(train_loss.item())
            train_acc_batch.append(train_acc.item())

            if device == "cpu":
                print('  Current batch: {} - Loss: {} - Acc: {}'.format(
                    step,
                    round(train_loss.item(), fraction),
                    round(train_acc.item(), fraction),
                ))
            elif device == "cuda":
                print('  Current batch: {} - Loss: {} - Acc: {}  AllocMem (Mb): {}'.format(
                    step,
                    round(train_loss.item(), fraction),
                    round(train_acc.item(), fraction),
                    round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                ))

        # get averages of loss and acc for epoch
        train_avg_loss_epoch = sum(train_loss_batch) / len(train_loss_batch)
        train_avg_acc_epoch = sum(train_acc_batch) / len(train_acc_batch)

        # save the best values
        if train_avg_loss_epoch < train_best_loss:
            train_best_loss = train_avg_loss_epoch
            train_best_loss_epoch = epoch
        if train_avg_acc_epoch > train_best_acc:
            train_best_acc = train_avg_acc_epoch
            train_best_acc_epoch = epoch

        print(' Current epoch: {} - avg. train Loss: {} - avg. train Acc: {}'.format(
            epoch,
            round(train_avg_loss_epoch, fraction),
            round(train_avg_acc_epoch, fraction)
        ))

        # release
        del train_preds
        del train_loss
        del train_acc
        del classes

        train_loss_epoch.append(train_avg_loss_epoch)
        train_acc_epoch.append(train_avg_acc_epoch)

        val_loss_batch = []
        val_acc_batch = []

        step = 0

        for valid_x, valid_y in input_valid_dl:

            step += 1

            # data to gpu
            valid_x = valid_x.to(device)
            valid_y = valid_y.to(device)

            # get values
            val_preds = model(valid_x)

            # get loss
            val_loss = input_loss_fn(val_preds, valid_y)

            if BINARY:
                classes = (val_preds > 0.5).cpu().detach().numpy()
            else:
                classes = np.argmax(val_preds.cpu().detach().numpy(), axis=1)

            val_acc = input_acc_fn(classes, valid_y.cpu().detach().numpy())

            # save the values
            val_loss_batch.append(val_loss.item())
            val_acc_batch.append(val_acc.item())

            if device == "cpu":
                print('  Current batch: {} - Loss: {} - Acc: {}'.format(
                    step,
                    round(val_loss.item(), fraction),
                    round(val_acc.item(), fraction)
                ))
            elif device == "cuda":
                print('  Current batch: {} - Loss: {} - Acc: {}  AllocMem (Mb): {}'.format(
                    step,
                    round(val_loss.item(), fraction),
                    round(val_acc.item(), fraction),
                    round(torch.cuda.memory_allocated() / 1024 / 1024, 4)
                ))

        # get averages of loss and acc for epoch
        val_avg_loss_epoch = sum(val_loss_batch) / len(val_loss_batch)
        val_avg_acc_epoch = sum(val_acc_batch) / len(val_acc_batch)

        if val_avg_loss_epoch < val_best_loss:
            val_best_loss = val_avg_loss_epoch
            val_best_loss_epoch = epoch
            # cpu_dict = {k:v.to('cpu') for k, v in model.state_dict().items()}
            best_model_dict = deepcopy(model.state_dict())

        if val_avg_acc_epoch > val_best_acc:
            val_best_acc = val_avg_acc_epoch
            val_best_acc_epoch = epoch

        print(' Current epoch: {} - avg. val Loss: {} - avg. val Acc: {}'.format(
            epoch,
            round(val_avg_loss_epoch, fraction),
            round(val_avg_acc_epoch, fraction)
        ))

        # release
        del val_preds
        del val_loss
        del val_acc
        del classes

        val_loss_epoch.append(val_avg_loss_epoch)
        val_acc_epoch.append(val_avg_acc_epoch)

        if EARLY_STOPPING > 0:
            if epoch - val_best_loss_epoch >= EARLY_STOPPING:
                print("Early stopping as no improvement in loss in {} rounds".format(EARLY_STOPPING))
                model.load_state_dict(best_model_dict)
                break

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    data = {
        "loss_train": train_loss_epoch,
        "acc_train": train_acc_epoch,
        "loss_val": val_loss_epoch,
        "acc_val": val_acc_epoch,
        "final_epoch": val_best_loss_epoch + 1
    }

    return data


gray = [150, 149, 158]

# set colors for the plots
colors = [(150 / 255, 149 / 255, 158 / 255),
          (230 / 255, 230 / 255, 235 / 255), (46 / 255, 45 / 255, 46 / 255),
          (7 / 255, 29 / 255, 232 / 255), (25 / 255, 227 / 255, 224 / 255),
          (186 / 255, 39 / 255, 32 / 255), (224 / 255, 7 / 255, 224 / 255)]
limits = [1, 2, 3, 4, 5, 6, 7, 50]
cmapOutput, normOutput = from_levels_and_colors(limits, colors)


def create_overview(data, plot, save, title=""):
    if plot is False and save is False:
        return

    fig, ax = plt.subplots(4, 4)

    i, j = 0, 0

    for elem in data:

        elem_x = elem[0]
        elem_y = elem[1]

        elem_x = elem_x.numpy()[0]
        elem_y = elem_y.numpy()

        ax[i, j].imshow(elem_x, cmap="gray")
        j += 1
        ax[i, j].imshow(elem_y, cmap=cmapOutput)

        j += 1

        if j == 4:
            i += 1
            j = 0

        if i == 4:
            break

    if title != "":
        plt.suptitle(title)

    if save:
        plt.savefig(OVERVIEW_FOLDER + "/" + title + "_" + MODEL_NAME + ".png")

    if plot:
        plt.show()


def create_graphs(data_dict, plot, save):
    if plot is False and save is False:
        return

    fix, ax_loss = plt.subplots()

    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")

    ax_acc = ax_loss.twinx()
    ax_acc.set_ylabel("Accuracy")

    ax_loss.xaxis.get_major_locator().set_params(integer=True)
    ax_acc.xaxis.get_major_locator().set_params(integer=True)

    counter = 0
    for key, elem in data_dict.items():

        if key.endswith("val"):
            if key.startswith("loss"):
                color = "red"
            if key.startswith("acc"):
                color = "blue"
        if key.endswith("train"):
            if key.startswith("loss"):
                color = "indianred"
            if key.startswith("acc"):
                color = "cornflowerblue"

        if key.startswith("loss"):
            ax_loss.plot(elem, label=key, color=color)
        if key.startswith("acc"):
            ax_acc.plot(elem, label=key, color=color)
        counter += 1

    ax_loss.legend(loc="upper left")
    ax_acc.legend(loc="upper right")

    if EARLY_STOPPING > 0 and len(next(iter(data_dict.values()))) != NR_EPOCHS:
        nr_epochs = len(next(iter(data_dict.values()))) - 1
        early_stopping_pos = nr_epochs - EARLY_STOPPING
        # get nr of epochs
        plt.axvline(x=early_stopping_pos, color='red', ls="--")

    if save:
        plt.savefig(GRAPH_FOLDER + "/" + MODEL_NAME + ".png")

    if plot:
        plt.show()


def create_tests(model, data, plot, save):
    if plot is False and save is False:
        return

    i = 0

    for data_x, data_y in data:

        # data to device
        data_x = data_x.to(device)
        data_y = data_y.to(device)

        # get values
        preds = model(data_x)

        if BINARY:
            classes = (preds > 0.5).cpu().detach().numpy()[0, 0, :, :]
        else:
            classes = np.argmax(preds.cpu().detach().numpy(), axis=1)[0, :, :]

        # convert data
        data_x = data_x.cpu()[0, 0, :, :]
        data_y = data_y.cpu()[0, :, :]

        fig, ax = plt.subplots(1, 3)

        ax[0].imshow(data_x, cmap="gray")
        ax[0].set_title("Image")

        ax[1].imshow(data_y)
        ax[1].set_title("Ground truth")

        ax[2].imshow(classes)
        ax[2].set_title("Prediction")

        if save:
            plt.savefig(TEST_FOLDER + "/" + MODEL_NAME + "_" + str(i) + ".png")

        if plot:
            plt.show()

        i = i + 1


def create_stats(nr_images_dict, data_dict):
    if save_stats is False:
        return

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    fields = [
        MODEL_NAME,
        dt_string,
        str(nr_images_dict[0]),
        str(train_perc),
        str(nr_images_dict[1]),
        str(val_perc),
        str(nr_images_dict[2]),
        str(test_perc),
        str(nr_images_dict[3]),
        str(IMG_SIZE[0]) + "_" + str(IMG_SIZE[1]),
        str(edge),
        str(NR_EPOCHS),
        str(data_dict["final_epoch"]),
        str(LEARNING_RATE),
        str(BATCH_SIZE),
        str(EARLY_STOPPING),
        LOSS_TYPE,
        str(KERNEL_SIZE),
        str(OUTPUT_LAYERS),
        str(round(data_dict["loss_train"][-1], 5)),
        str(round(data_dict["acc_train"][-1], 5)),
        str(round(data_dict["loss_val"][-1], 5)),
        str(round(data_dict["acc_val"][-1], 5))
    ]

    with open(SAVE_FOLDER + '/_statistics.csv', 'a', newline='') as fd:
        writer = csv.writer(fd, delimiter=";")
        writer.writerow(fields)


if __name__ == "__main__":

    device = None
    if DEVICE == "cpu":
        device = 'cpu'
    elif DEVICE == "gpu":
        device = 'cuda'
    elif DEVICE == "auto":
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    filesList = get_file_list(MASK_FOLDER)

    # -- loading data -- #
    print('Load Data:')
    x = load_data(IMG_FOLDER, filesList, IMG_SIZE, remove_edge=True, expand=True, random=True, input_augment=augment)
    y = load_data(MASK_FOLDER, filesList, IMG_SIZE, random=True, input_augment=augment, is_mask=True)

    print(x.shape)
    print(y.shape)

    # get number of images
    number_of_images = x.shape[0]

    #the classes on the label begin with 1 and not with 0
    if BINARY == False:
        y = y - 1

    # get max val of y to check if output neurons are set right
    y_max = np.amax(y)
    if BINARY:
        if OUTPUT_LAYERS > 1:
            print("For the specified settings the number of output layers must be {}".format(1))
            exit()
    else:
        if OUTPUT_LAYERS != y_max:
            print("For the specified settings the number of output layers must be {}".format(y_max))
            exit()

    # for some losses y must have another dimension
    if BINARY and LOSS_TYPE == "crossentropy":
        y = np.expand_dims(y, 1)

    # convert data to tensor
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    # convert the data types
    x = x.float()
    if BINARY and LOSS_TYPE == "crossentropy":
        y = y.float()
    else:
        y = y.long()

    print('-' * 10)

    # -- prepare training --#
    print("Prepare data:")

    # split data
    percentages = [train_perc, val_perc, test_perc]
    train_x, val_x, test_x = split_data(x, percentages)
    train_y, val_y, test_y = split_data(y, percentages)

    num_train_images = train_x.shape[0]
    num_val_images = val_x.shape[0]
    num_test_images = test_x.shape[0]

    nrImagesDict = [number_of_images, num_train_images, num_val_images, num_test_images]

    # put data into datasets
    train_ds = TensorDataset(train_x, train_y)
    valid_ds = TensorDataset(val_x, val_y)
    test_ds = TensorDataset(test_x, test_y)

    # see data
    create_overview(train_ds, plot_overview, save_overview, title="training")
    create_overview(valid_ds, plot_overview, save_overview, title="validation")

    # create dataloaders
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    test_dl = DataLoader(test_ds, batch_size=1)

    # specify the model
    unet = UNET(1, OUTPUT_LAYERS, binary=BINARY)

    # specify the loss
    loss_fn = None
    if LOSS_TYPE == "crossentropy":
        if BINARY:
            loss_fn = nn.BCELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
    elif LOSS_TYPE == "focal":
        if BINARY:
            loss_fn = BinaryFocalLoss()
        else:
            loss_fn = MultiFocalLoss()

    # specify the optimizer
    optim = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)

    # train the model
    dataDict = train(unet, train_dl, valid_dl, optim, loss_fn, acc_fn, epochs=NR_EPOCHS)

    # evaluate the model
    create_graphs(dataDict, plot_graphs, save_graphs)

    # visualize the tests
    create_tests(unet, test_dl, plot_test, save_test)

    create_stats(nrImagesDict, dataDict)

    # save model
    if save_model:
        torch.save(unet, SAVE_FOLDER + "/" + MODEL_NAME + ".pt")
