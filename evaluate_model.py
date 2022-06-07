import matplotlib.pyplot as plt
import numpy as np

import load_data_from_json as ldfj

model_name = "<Your model id>"
model_folder = "<Enter your path to the folder where the models should be stored>"

evaluation_params = {
    "show_train": True,
    "show_val": True,
    # what to display
    "display_loss": True,
    "save_loss": True,
    "display_accuracy": True,
    "save_accuracy": True,
    "display_f1": True,
    "save_f1": True,
    # how to display
    "show_title": True,
    "show_axis_labels": True,
    "show_line_labels": True,
    "show_values": False,
    "show_trend": True
}


def evaluate_model(json_name, json_folder, params):

    assert params["show_train"] or params["show_val"]

    assert params["show_values"] or params["show_trend"]

    # build path
    json_path = json_folder + "/" + json_name + ".json"

    # load the json file
    data = ldfj.load_data_from_json(json_path)

    if data is None:
        print(f"Something went wrong with loading file at '{json_path}'")

    def build_loss(params2, statistics, polynomial=3):

        train_loss = statistics["train_loss"]
        val_loss = statistics["val_loss"]

        # dict to list
        train_loss = list(train_loss.values())
        val_loss = list(val_loss.values())

        fig, ax = plt.subplots(1)

        if params2["show_title"]:
            ax.set_title("Development of Loss")

        if params2["show_trend"]:

            # get y values
            epochs = list(range(len(train_loss)))

            # create trend-line for train
            z_train = np.polyfit(epochs, train_loss, polynomial)
            p_train = np.poly1d(z_train)

            # plot trend line for train
            trend_train, = ax.plot(epochs, p_train(epochs), "b--", label="train trend")
            trend_train.set_color("lightskyblue")

            # create trend-line for val
            z_val = np.polyfit(epochs, val_loss, polynomial)
            p_val = np.poly1d(z_val)

            # plot trend-line for val
            trend_val, = ax.plot(epochs, p_val(epochs), "b--", label="val trend")
            trend_val.set_color("coral")

        if params2["show_axis_labels"]:
            ax.set_xlabel('Epochs')
            ax.set_ylabel("Loss per Epoch")

        if params2["show_values"]:
            ax.plot(train_loss, label="train loss")
            ax.plot(val_loss, label="val loss")

        if params2["show_line_labels"]:
            plt.legend()

        plt.show()

    def build_accuracy(params2, statistics):

        train_accuracy = statistics["train_acc"]
        val_accuracy = statistics["val_acc"]

        # dict to list
        train_accuracy = list(train_accuracy.values())
        val_accuracy = list(val_accuracy.values())

        fig, ax = plt.subplots(1)

        if params2["show_title"]:
            ax.set_title("Development of Accuracy")

        if params2["show_trend"]:

            # get y values
            epochs = list(range(len(train_accuracy)))

            # create trend-line
            z_train = np.polyfit(epochs, train_accuracy, 1)
            p_train = np.poly1d(z_train)

            trend_train, = ax.plot(epochs, p_train(epochs), "b--", label="train trend")
            trend_train.set_color("lightskyblue")

            z_val = np.polyfit(epochs, val_accuracy, 1)
            p_val = np.poly1d(z_val)

            trend_val, = ax.plot(epochs, p_val(epochs), "b--", label="val trend")
            trend_val.set_color("coral")

        if params2["show_axis_labels"]:
            ax.set_xlabel('Epochs')
            ax.set_ylabel("Accuracy per Epoch")

        if params2["show_values"]:
            ax.plot(train_accuracy, label="train accuracy")
            ax.plot(val_accuracy, label="val accuracy")

        if params2["show_line_labels"]:
            plt.legend()

        plt.show()

    def build_f1_score(params2, statistics):
        train_accuracy = statistics["train_f1"]
        val_accuracy = statistics["val_f1"]

        # dict to list
        train_accuracy = list(train_accuracy.values())
        val_accuracy = list(val_accuracy.values())

        fig, ax = plt.subplots(1)

        if params2["show_title"]:
            ax.set_title("Development of F1-Score")

        if params2["show_trend"]:

            # get y values
            epochs = list(range(len(train_accuracy)))

            # create trend-line
            z_train = np.polyfit(epochs, train_accuracy, 1)
            p_train = np.poly1d(z_train)

            trend_train, = ax.plot(epochs, p_train(epochs), "b--", label="train trend")
            trend_train.set_color("lightskyblue")

            z_val = np.polyfit(epochs, val_accuracy, 1)
            p_val = np.poly1d(z_val)

            trend_val, = ax.plot(epochs, p_val(epochs), "b--", label="val trend")
            trend_val.set_color("coral")

        if params2["show_axis_labels"]:
            ax.set_xlabel('Epochs')
            ax.set_ylabel("F1-Score per Epoch")

        if params2["show_values"]:
            ax.plot(train_accuracy, label="train F1-score")
            ax.plot(val_accuracy, label="val F1-score")

        if params2["show_line_labels"]:
            plt.legend()

        plt.show()

    if params["display_loss"] or params["save_loss"]:
        build_loss(params, data["statistics"])

    if params["display_accuracy"] or params["save_accuracy"]:
        build_accuracy(params, data["statistics"])

    if params["display_f1"] or params["save_f1"]:
        build_f1_score(params, data["statistics"])


if __name__ == "__main__":


    evaluate_model(model_name, folder, evaluation_params)
