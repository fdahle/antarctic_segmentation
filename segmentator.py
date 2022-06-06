import sys
import os
import copy
import time
import random
import numpy as np
import PIL.Image
import cv2
import subprocess as spc
from datetime import datetime

from tkinter import *
from tkinter import ttk
from tkinter import font
from tkinter import scrolledtext
from tkinter import messagebox

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import from_levels_and_colors
import matplotlib.patches as patches

from skimage.draw import line
from scipy.signal import convolve2d

import cut_off_edge as coe
import get_ids_from_folder as giff

import sub_functions.set_class as sc
import sub_functions.remove_small_clusters as rsc
from sub_functions.segment import segment

segmentator_params = {
    "min_cluster_size": 50000,  # the minimum size of a cluster so that the cluster is not removed
    "scribble_size": 25,  # the initial size of the scribbler
    "small_image_size": 3000,
    "path_images": "<Enter your path to the folder with the images>",  # path to the raw images
    "path_segmented": "<Enter your path to the folder with the segmented images>"  # path to the unsupervised images
}


class Segmentator:

    def __init__(self, params):
        # params for Segmentator
        self.params = params

        # params for the gui
        self.window = Tk()
        self.window.title('Segmentator')
        self.window.geometry('1600x1000+10+10')  # window size
        self.window.protocol("WM_DELETE_WINDOW", self.stop_window)  # exit program when quitting
        self.frame = Frame(self.window, relief='sunken')
        self.frame.pack(fill=BOTH, expand=True, padx=10, pady=20)

        # run time params
        self.list_of_files = []  # list of ids of all possible files that can be selected
        self.selected_file_id = None  # which is the currently selected id
        self.img_orig_dims = None
        self.border_bounds = None
        self.bool_image_locked = False  # flags if currently an operation is going on and therefore the image is locked
        # for other operations
        self.bool_mouse_left_click = False  # flags if the left button of the mouse is currently clicked
        self.bool_image_loaded = False  # if false no image is loaded and many processes will not work
        self.bool_scribble_mode = False  # flags if currently a scribble is drawing
        self.bool_scribble_in_image = False  # flags that state if there's a scribble in an image
        self.bool_box_drawing_mode = False  # flags if currently a box (for resegmenting) is  drawn
        self.bool_box_in_image = False  # flags if there's a box in the image

        # run time variables for scribble and box
        self.scribble_x = None
        self.scribble_y = None
        self.box_top_y = None
        self.box_bottom_y = None
        self.box_left_x = None
        self.box_right_x = None

        # user settings
        self.scribble_type = IntVar()  # 1 for draw, 2 for click
        self.bool_cluster_images = BooleanVar(self.window, True)  # should images be clustered when loading or not
        self.bool_overwrite_classes = BooleanVar(self.window, False)  # if true, assigned classes can be reassigned
        self.bool_remove_scribbles_after_change = BooleanVar(self.window, True)

        # params that will stay consistent
        self.classes = ["not set", "Ice", "Snow", "Rocks", "Water", "Clouds", "Sky", "Unknown"]
        self.random_cmap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
        custom_colors = [(242 / 255, 170 / 255, 217 / 255), (150 / 255, 149 / 255, 158 / 255),
                         (230 / 255, 230 / 255, 235 / 255), (46 / 255, 45 / 255, 46 / 255),
                         (7 / 255, 29 / 255, 232 / 255), (25 / 255, 227 / 255, 224 / 255),
                         (186 / 255, 39 / 255, 32 / 255), (224 / 255, 7 / 255, 224 / 255)]
        limits = range(0, 9)
        self.cmapOutput, self.normOutput = from_levels_and_colors(limits, custom_colors)

        # image variables
        img_size = params["small_image_size"]
        self.image_unsupervised = np.zeros([img_size, img_size])  # the unsupervised image
        self.image_unsupervised_original = None  # original that will never be touched (need for reset)
        self.image_raw = np.zeros([img_size, img_size])  # the raw image
        self.image_final = np.zeros([img_size, img_size])  # the final image
        self.image_clustered = None  # a clustered image that is stored temporary to make "set_class" faster
        self.image_unsupervised_backup = None  # backup of unsupervised image (for undo)
        self.image_final_backup = None  # backup of final image (for undo)

        # init all gui elements so that they can later be called within the complete class
        self.string_var_image_selector = None
        self.label_selected_image = None
        self.text_scribble_size = None
        self.text_x_input = None
        self.text_y_input = None
        self.text_box_number = None
        self.button_switch_scribble = None
        self.combo_class = None
        self.tree_supervised = None
        self.string_var_x_pos = None
        self.string_var_y_pos = None
        self.string_var_unsupervised_class = None
        self.string_var_supervised_class = None
        self.fig_unsupervised = None
        self.ax_unsupervised = None
        self.im_unsupervised = None
        self.canvas_unsupervised = None
        self.fig_raw = None
        self.ax_raw = None
        self.im_raw = None
        self.canvas_raw = None
        self.im_unsupervised = None
        self.fig_overlay = None
        self.ax_overlay = None
        self.im_overlay1 = None
        self.im_overlay2 = None
        self.canvas_overlay = None
        self.fig_final = None
        self.ax_final = None
        self.im_final = None
        self.canvas_final = None
        self.scrolled_text_log = None

        # stuff that must be done when starting the program
        self.init_build_list_of_files()
        self.init_build_gui()

    # this function is called at the beginning and runs the main loop so that the window is visible
    def display_window(self):
        self.frame.mainloop()

    # this function is called when closing the window and closes everything
    def stop_window(self):
        self.window.quit()
        self.window.destroy()
        exit()

    # this function is called at the beginning and checks which files can be used for segmenting
    def init_build_list_of_files(self):
        raw_path = self.params["path_images"]
        segmented_base_path = self.params["path_segmented"]

        segmented_unsupervised_path = segmented_base_path + "/unsupervised"

        # build list of folders to check
        folders = [
            raw_path,
            segmented_unsupervised_path
        ]

        # get the ids that exist in both folders
        ids = giff.get_ids_from_folder(folders)

        # stop complete program if no files are there
        if len(ids) == 0:
            print("No files could be found at {} with unsupervised labels at {}".format(
                raw_path, segmented_base_path))
            exit()

        self.list_of_files = ids

    # this function is called at the beginning and build all elements of the gui
    def init_build_gui(self):
        #
        # image selection ###
        #

        # image selector
        self.string_var_image_selector = StringVar(self.frame)
        self.string_var_image_selector.trace("w", self.event_load_images_to_gui)
        image_selector = OptionMenu(self.frame, self.string_var_image_selector, *self.list_of_files)
        image_selector.grid(row=0, column=0, columnspan=3)

        # random button
        button_random_img = Button(self.frame, text="Random", command=self.event_get_random_image)
        button_random_img.grid(row=0, column=3, columnspan=2)

        # label for selected image and link to open
        self.label_selected_image = Label(self.frame, fg="blue", cursor="hand2")
        _f = font.Font(self.label_selected_image, self.label_selected_image.cget("font"))
        _f.configure(underline=True)
        self.label_selected_image.configure(font=_f)
        self.label_selected_image.config(text=self.selected_file_id)
        self.label_selected_image.bind("<Button-1>", lambda e: self.event_show_selected_image())
        self.label_selected_image.grid(row=0, column=5, columnspan=3)

        # failed button
        button_failed_img = Button(self.frame, text="Failed", command=self.event_set_image_failed)
        button_failed_img.grid(row=0, column=8, columnspan=2)

        ###
        # image clustering
        ###

        # check if clustering should be done
        check_clustering = Checkbutton(self.frame, text="cluster images at loading", variable=self.bool_cluster_images)
        check_clustering.grid(row=1, column=3)

        # label for min cluster size
        string_var_min_cluster_size = StringVar(self.frame)
        string_var_min_cluster_size.set("minClusterSize")
        label_min_cluster_size = Label(self.frame, textvariable=string_var_min_cluster_size)
        label_min_cluster_size.grid(row=1, column=5, sticky="E")

        # entry for min cluster size
        text_min_cluster_size = Entry(self.frame, width=6)
        text_min_cluster_size.insert(0, self.params["min_cluster_size"])
        text_min_cluster_size.grid(row=1, column=6, columnspan=2)

        # button for clustering
        button_recluster = Button(self.frame, text="Recluster", command=self.event_recluster_image)
        button_recluster.grid(row=1, column=8, columnspan=2)

        ###
        # Scribble
        ###

        # label for scribble size
        string_var_scribble_size = StringVar(self.frame)
        string_var_scribble_size.set("ScribbleSize")
        label_scribble_size = Label(self.frame, textvariable=string_var_scribble_size)
        label_scribble_size.grid(row=5, column=2)

        # entry for scribble size
        self.text_scribble_size = Entry(self.frame, width=3)
        self.text_scribble_size.insert(0, self.params["scribble_size"])
        self.text_scribble_size.grid(row=5, column=3, columnspan=2)

        # set default scribble to draw
        self.scribble_type.set(1)

        # radio buttons to set the scribble type
        radio_scribble_1 = Radiobutton(self.frame, text="draw", variable=self.scribble_type, value=1)
        radio_scribble_2 = Radiobutton(self.frame, text="click", variable=self.scribble_type, value=2)

        radio_scribble_1.grid(row=5, column=5)
        radio_scribble_2.grid(row=5, column=6)

        # scribble button
        self.button_switch_scribble = Button(self.frame, text="Scribble", command=self.event_switch_scribble_mode)
        self.button_switch_scribble.grid(row=5, column=7, columnspan=2)

        # clear scribble button
        button_clear_scribble = Button(self.frame, text="Clear", command=self.event_clear_scribble)
        button_clear_scribble.grid(row=5, column=9, columnspan=2)

        ###
        # resegment
        ###

        # draw box button
        button_draw_box = Button(self.frame, text="Draw Box", command=self.event_draw_resegmenting_box)
        button_draw_box.grid(row=7, column=1, columnspan=2)

        # delete box button
        button_delete_box = Button(self.frame, text="Delete Box", command=self.event_delete_resegmenting_box)
        button_delete_box.grid(row=7, column=3, columnspan=2)

        # label for number of segments
        string_var_box_number = StringVar(self.frame)
        string_var_box_number.set("Nr. of Segments")
        label_box_number = Label(self.frame, textvariable=string_var_box_number)
        label_box_number.grid(row=7, column=6, sticky="E")

        # text for number of segments
        self.text_box_number = Text(self.frame, height=1, width=3)
        self.text_box_number.grid(row=7, column=7)

        # button to resegment
        button_resegment = Button(self.frame, text="Resegment", command=self.event_resegment_box)
        button_resegment.grid(row=7, column=8)

        ###
        # other options
        ###

        # option for overwrite
        check_overwrite = Checkbutton(self.frame, text="overwrite", variable=self.bool_overwrite_classes)
        check_overwrite.grid(row=8, column=3)

        check_remove_scribbles = Checkbutton(self.frame, text="autoremove scribbles",
                                             variable=self.bool_remove_scribbles_after_change)
        check_remove_scribbles.grid(row=8, column=4)

        # undo button
        button_undo = Button(self.frame, text="Undo", command=self.event_undo_last_action)
        button_undo.grid(row=8, column=7, columnspan=2)

        ###
        # setting the class
        ###

        # input for x
        string_var_x_input = StringVar(self.frame)
        string_var_x_input.set("x:")
        label_x_input = Label(self.frame, textvariable=string_var_x_input)
        label_x_input.grid(row=11, column=0, sticky="E")

        self.text_x_input = Text(self.frame, height=1, width=5)
        self.text_x_input.grid(row=11, column=1, columnspan=2)

        # input for y
        string_var_y_input = StringVar(self.frame)
        string_var_y_input.set("y:")
        label_y_input = Label(self.frame, textvariable=string_var_y_input)
        label_y_input.grid(row=11, column=3, sticky="E")

        self.text_y_input = Text(self.frame, height=1, width=5)
        self.text_y_input.grid(row=11, column=4, columnspan=2)

        # custom classes without first entry 'not set'
        classes = copy.deepcopy(self.classes)
        del classes[0]

        # option field for classes
        self.combo_class = ttk.Combobox(self.frame, width=27)
        self.combo_class['values'] = classes
        self.combo_class.grid(row=11, column=6, columnspan=2)

        # Button set class for one cluster
        button_set_class = Button(self.frame, text="Set Class", command=self.event_set_class)
        button_set_class.grid(row=11, column=8, columnspan=2)

        # Button set class for all
        button_set_all = Button(self.frame, text="Set All", command=self.event_set_class_all)
        button_set_all.grid(row=11, column=10, columnspan=2)

        #
        # tree for image information
        #

        var_tree_supervised = StringVar()
        var_tree_supervised.set("final classes")
        label_tree_supervised = Label(self.frame, textvariable=var_tree_supervised)
        label_tree_supervised.grid(row=12, column=0, columnspan=13, sticky="S")

        self.tree_supervised = ttk.Treeview(self.frame, columns=('SegmentId', 'Percentages', 'Count'), show='headings')
        self.tree_supervised.grid(row=13, column=0, columnspan=13, rowspan=3)

        # define headings
        self.tree_supervised.heading('SegmentId', text='Class')
        self.tree_supervised.heading('Percentages', text='Pixels')
        self.tree_supervised.heading('Count', text='Percentage')

        #
        # final buttons
        #

        # reset button
        button_reset = Button(self.frame, text="Reset", command=self.event_reset_image)
        button_reset.grid(row=21, column=1, columnspan=2)

        # save button
        button_save = Button(self.frame, text="Save", command=self.event_save_image)
        button_save.grid(row=21, column=11, columnspan=2)

        #
        # image information
        #

        self.string_var_x_pos = StringVar(self.frame)
        self.string_var_x_pos.set("x: ...")
        label_x_pos = Label(self.frame, textvariable=self.string_var_x_pos)
        label_x_pos.grid(row=0, column=20)

        self.string_var_y_pos = StringVar(self.frame)
        self.string_var_y_pos.set("y: ...")
        label_y_pos = Label(self.frame, textvariable=self.string_var_y_pos)
        label_y_pos.grid(row=0, column=22)

        self.string_var_unsupervised_class = StringVar(self.frame)
        self.string_var_unsupervised_class.set("class: ...")
        label_unsupervised_class = Label(self.frame, textvariable=self.string_var_unsupervised_class)
        label_unsupervised_class.grid(row=0, column=24)

        self.string_var_supervised_class = StringVar(self.frame)
        self.string_var_supervised_class.set("class: ...")
        label_supervised_class = Label(self.frame, textvariable=self.string_var_supervised_class)
        label_supervised_class.grid(row=0, column=26)

        #
        # images
        #

        figsize = (4, 4)

        # unsupervised image
        self.fig_unsupervised = Figure(figsize=figsize)
        self.ax_unsupervised = self.fig_unsupervised.add_subplot(111)
        self.im_unsupervised = self.ax_unsupervised.imshow(self.image_unsupervised,
                                                           cmap=self.random_cmap,
                                                           interpolation="None")

        self.canvas_unsupervised = FigureCanvasTkAgg(self.fig_unsupervised, master=self.frame)
        self.canvas_unsupervised.draw()
        self.canvas_unsupervised.get_tk_widget().grid(row=1, column=15, rowspan=8, columnspan=8,
                                                      padx=5, pady=5)

        # raw image
        self.fig_raw = Figure(figsize=figsize)
        self.ax_raw = self.fig_raw.add_subplot(111)
        self.im_raw = self.ax_raw.imshow(self.image_raw, cmap="gray", vmin=0, vmax=255)

        self.canvas_raw = FigureCanvasTkAgg(self.fig_raw, master=self.frame)
        self.canvas_raw.draw()
        self.canvas_raw.get_tk_widget().grid(row=1, column=24, rowspan=8, columnspan=8,
                                             padx=5, pady=5)

        # final image
        self.fig_final = Figure(figsize=figsize)
        self.ax_final = self.fig_final.add_subplot(111)
        self.im_final = self.ax_final.imshow(self.image_final, cmap=self.cmapOutput,
                                             norm=self.normOutput, interpolation="None")
        self.canvas_final = FigureCanvasTkAgg(self.fig_final, master=self.frame)
        self.canvas_final.draw()
        self.canvas_final.get_tk_widget().grid(row=10, column=15, rowspan=8, columnspan=8,
                                               padx=5, pady=5)

        # overlay image
        self.fig_overlay = Figure(figsize=figsize)
        self.ax_overlay = self.fig_overlay.add_subplot(111)
        self.im_overlay1 = self.ax_overlay.imshow(self.image_raw, cmap="gray", vmin=0, vmax=255)
        self.im_overlay2 = self.ax_overlay.imshow(self.image_final, cmap=self.cmapOutput,
                                                  norm=self.normOutput, interpolation='none', alpha=0.5)

        self.canvas_overlay = FigureCanvasTkAgg(self.fig_overlay, master=self.frame)
        self.canvas_overlay.draw()
        self.canvas_overlay.get_tk_widget().grid(row=10, column=24, rowspan=8, columnspan=8,
                                                 padx=5, pady=5)

        # remove the whitespace around the images
        self.fig_unsupervised.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.fig_raw.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.fig_final.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.fig_overlay.subplots_adjust(left=0, right=1, bottom=0, top=1)

        # remove the axis around the images
        self.ax_unsupervised.set_axis_off()
        self.ax_raw.set_axis_off()
        self.ax_final.set_axis_off()
        self.ax_overlay.set_axis_off()

        # add the events for motion
        self.fig_unsupervised.canvas.mpl_connect("motion_notify_event", self.event_mouse_hover)
        self.fig_raw.canvas.mpl_connect("motion_notify_event", self.event_mouse_hover)
        self.fig_final.canvas.mpl_connect("motion_notify_event", self.event_mouse_hover)
        self.fig_overlay.canvas.mpl_connect("motion_notify_event", self.event_mouse_hover)

        # add the events for mouse click
        self.fig_unsupervised.canvas.mpl_connect("button_press_event", self.event_mouse_click)
        self.fig_raw.canvas.mpl_connect("button_press_event", self.event_mouse_click)
        self.fig_final.canvas.mpl_connect("button_press_event", self.event_mouse_click)
        self.fig_overlay.canvas.mpl_connect("button_press_event", self.event_mouse_click)

        # add the events for mouse release
        self.fig_unsupervised.canvas.mpl_connect("button_release_event", self.event_mouse_release)
        self.fig_raw.canvas.mpl_connect("button_release_event", self.event_mouse_release)
        self.fig_final.canvas.mpl_connect("button_release_event", self.event_mouse_release)
        self.fig_overlay.canvas.mpl_connect("button_release_event", self.event_mouse_release)

        #
        # log
        #

        self.scrolled_text_log = scrolledtext.ScrolledText(self.frame, height=5, state='disabled')
        self.scrolled_text_log.grid(row=19, column=15, rowspan=3, columnspan=17)

    # load the images into the gui
    def event_load_images_to_gui(self, *_):

        self.bool_image_locked = True

        # get the id of the image that should be loaded
        file_id = self.string_var_image_selector.get()

        self.event_add_to_log("start loading {}".format(file_id))

        # reset possible boxes
        self.box_top_y = None
        self.box_bottom_y = None
        self.box_left_x = None
        self.box_right_x = None

        # remove patches
        [p.remove() for p in reversed(self.ax_unsupervised.patches)]

        # reset possible drawn segments
        self.image_final = np.zeros([self.params["small_image_size"], self.params["small_image_size"]])

        # load the images in Pillow
        pil_unsupervised: Image = PIL.Image.open(self.params["path_segmented"] + "/unsupervised/" + file_id + ".tif")
        pil_raw: Image = PIL.Image.open(self.params["path_images"] + "/" + file_id + ".tif")

        # everything to np
        image_unsupervised = np.array(pil_unsupervised)
        image_raw = np.array(pil_raw)

        # save the original image size without cutting of anything
        self.img_orig_dims = image_raw.shape

        # remove the borders of the images
        image_unsupervised, edge_dims = coe.cut_off_edge(image_unsupervised, img_id=file_id, cut_method="auto",
                                                         extra_edge=50, return_edge_dims=True)
        image_raw, edge_dims = coe.cut_off_edge(image_raw, img_id=file_id, cut_method="auto", extra_edge=50,
                                                return_edge_dims=True)

        # save the border bounds (what is removed from the images on each side)
        self.border_bounds = edge_dims

        # change the unsupervised images (remove small clusters and add new ids)
        if self.bool_cluster_images.get():
            self.event_add_to_log("start reclustering of {}".format(file_id))
            image_unsupervised = rsc.remove_small_clusters(image_unsupervised,
                                                           self.params["min_cluster_size"])
            self.event_add_to_log("reclustering of {} finished successfully".format(file_id))

        # resize so that displaying is way faster
        self.image_raw = cv2.resize(image_raw,
                                    dsize=(self.params["small_image_size"],
                                           self.params["small_image_size"]),
                                    interpolation=cv2.INTER_NEAREST)
        self.image_unsupervised = cv2.resize(image_unsupervised,
                                             dsize=(self.params["small_image_size"],
                                                    self.params["small_image_size"]),
                                             interpolation=cv2.INTER_NEAREST)

        # save the backup
        self.image_unsupervised_original = copy.deepcopy(image_unsupervised)

        # update the plots
        self.event_update_plots()

        # update the link to image
        self.label_selected_image.config(text=file_id)

        # update the tree
        self.event_refresh_tree()

        self.selected_file_id = file_id
        self.event_add_to_log("{} loaded successfully".format(file_id))
        self.bool_image_loaded = True
        self.bool_image_locked = False

    # load a random image from the list of files
    def event_get_random_image(self):

        while True:

            # select a random id
            random_image_id = random.choice(self.list_of_files)

            # if there's only one image in the list we don't need to repeat the random search
            if len(self.list_of_files) == 1:
                break

            # do until we get a new random image
            if random_image_id != self.selected_file_id:
                break

        # set the random id to the selector so that the loading process can start
        self.string_var_image_selector.set(random_image_id)

    # update the matplotlib plots
    def event_update_plots(self):

        # get max val for image in order to update vmax
        amax = np.amax(self.image_unsupervised)
        if amax < 100:
            amax = 100

        self.im_unsupervised.set_clim(vmin=100, vmax=amax)

        self.im_unsupervised.set_data(self.image_unsupervised)
        self.im_raw.set_data(self.image_raw)
        self.im_final.set_data(self.image_final)
        self.im_overlay1.set_data(self.image_raw)
        self.im_overlay2.set_data(self.image_final)

        self.canvas_unsupervised.draw_idle()
        self.canvas_raw.draw_idle()
        self.canvas_final.draw_idle()
        self.canvas_overlay.draw_idle()

    # set the input for the class tree
    def event_refresh_tree(self):

        # get the values
        unique, counts = np.unique(self.image_final, return_counts=True)

        # calc percentages
        total = np.sum(counts)
        percentages = np.round(counts / total * 100, 2)

        # merge everything
        val_dict = dict(zip(unique, list(zip(counts, percentages))))

        # delete content of tree
        self.tree_supervised.delete(*self.tree_supervised.get_children())

        # TODO
        # if 0 in val_dict:
        #    self.tree_supervised.insert("", "end", values=("not set", val_dict[0][0], val_dict[0][1]))

        # add values to tree
        for i, elem in enumerate(self.classes):
            if i in val_dict:
                self.tree_supervised.insert("", "end", values=(elem, val_dict[i][0], val_dict[i][1]))

    # if an unsupervised image is not good, set it to failed
    def event_set_image_failed(self):

        # check if image is locked
        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        # reset image values
        img_size = self.params["small_image_size"]
        self.image_unsupervised = np.zeros([img_size, img_size])  # the unsupervised image
        self.image_unsupervised_original = None  # original that will never be touched (need for reset)
        self.image_raw = np.zeros([img_size, img_size])  # the raw image
        self.image_final = np.zeros([img_size, img_size])  # the final image
        self.image_clustered = np.zeros([img_size, img_size])  # clus. img that is stored temp. to make "set_class" fast
        self.image_unsupervised_backup = None  # backup of unsupervised image (for undo)
        self.image_final_backup = None  # backup of final image (for undo)

        self.event_update_plots()

        # get selected file and relocate it to the failed folder
        if self.selected_file_id in self.list_of_files:
            self.list_of_files.remove(self.selected_file_id)
            os.rename(self.params["path_segmented"] + "/unsupervised/" + self.selected_file_id + ".tif",
                      self.params["path_segmented"] + "/unsupervised/failed/" + self.selected_file_id + ".tif")
        else:
            self.bool_image_locked = False
            print("That should not happen.")

        # reset values
        self.selected_file_id = None
        self.bool_image_locked = False
        self.bool_image_loaded = False

        self.event_add_to_log("{} is set to failed".format(self.selected_file_id))

    # recluster an image (for example to try out a different min_cluster_size)
    def event_recluster_image(self, text_field):

        if self.selected_file_id is None:
            self.event_add_to_log("Please select first an image")
            return

        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        try:
            min_cluster_size = int(text_field.get())

            # in order to cause an exception if negative value
            if min_cluster_size < 0:
                min_cluster_size = min_cluster_size / 0

        except (Exception,):
            self.bool_image_locked = False
            print("please enter a positive int value")
            return

        start_time = time.time()
        self.image_unsupervised = rsc.remove_small_clusters(self.image_unsupervised_original, min_cluster_size)

        self.event_add_to_log("reclustering done in {} seconds".format(time.time() - start_time))

        self.bool_image_locked = False

    # switch between normal and scribble mode
    def event_switch_scribble_mode(self):

        if self.selected_file_id is None:
            self.event_add_to_log("Please select first an image")
            return

        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return

        # scribble mode active -> deactivate
        if self.bool_scribble_mode:

            self.bool_scribble_mode = False
            self.window.config(cursor="arrow")
            self.button_switch_scribble['text'] = 'Scribble'

            self.scribble_x = None
            self.scribble_y = None

        # scribble mode inactive -> activate
        else:

            # check if scribble size is correct
            try:
                self.params["scribble_size"] = int(self.text_scribble_size.get())
            except (Exception,):
                self.event_add_to_log("Error: Please enter a positive value for the Scribble size")
                return

            self.bool_scribble_mode = True
            self.window.config(cursor="crosshair")
            self.button_switch_scribble['text'] = 'Stop'

    # remove existing scribbles on the image by loading the stored original backup
    def event_clear_scribble(self):

        self.bool_image_locked = True

        # if there's a scribble in the image load the original to remove all scribbles
        if self.bool_scribble_in_image:
            self.image_unsupervised = copy.deepcopy(self.image_unsupervised_original)
            self.event_update_plots()

        self.bool_image_locked = False

    # initialize the drawing mode for box (and remove old box if available)
    def event_draw_resegmenting_box(self):

        if self.selected_file_id is None:
            self.event_add_to_log("Please select first an image")
            return

        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return

        self.bool_image_locked = True

        # if a box is already there delete the old box
        if self.box_top_y is not None and \
                self.box_bottom_y is not None and \
                self.box_left_x is not None and \
                self.box_right_x is not None:
            self.box_top_y = None
            self.box_bottom_y = None
            self.box_left_x = None
            self.box_right_x = None

        self.event_add_to_log("Draw box: first topLeft and then bottomRight")

        self.bool_box_drawing_mode = True
        self.window.config(cursor="crosshair")

    # delete an existing box or exit box mode
    def event_delete_resegmenting_box(self):

        if self.selected_file_id is None:
            self.event_add_to_log("Please select first an image")
            return

        if self.bool_image_locked and self.bool_box_drawing_mode is False:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return

        self.box_top_y = None
        self.box_bottom_y = None
        self.box_left_x = None
        self.box_right_x = None

        # remove patches
        [p.remove() for p in reversed(self.ax_unsupervised.patches)]
        self.fig_unsupervised.canvas.draw_idle()

        self.bool_image_locked = False
        self.bool_box_drawing_mode = False
        self.bool_box_in_image = False

        self.window.config(cursor="arrow")

    # resegment only a small subset
    def event_resegment_box(self):

        if self.selected_file_id is None:
            self.event_add_to_log("Please select first an image")
            return

        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return

        nr_clusters = self.text_box_number.get("1.0", END)
        try:
            nr_clusters = int(nr_clusters)
        except (Exception,):
            self.event_add_to_log("Error: Nr. of Clusters is not a valid number")
            self.bool_image_locked = False
            return

        if nr_clusters < 0:
            self.event_add_to_log("Error: Nr. of Clusters is not a valid number")
            self.bool_image_locked = False
            return

        if self.box_top_y is None or self.box_bottom_y is None or \
                self.box_left_x is None or self.box_right_x is None:
            self.event_add_to_log("Error: Something went wrong. Please redraw the box")
            self.bool_image_locked = False
            return

        edge = 20

        # increase the subset so that we can remove it afterwards again (edges of class. are always bad)
        subset = self.image_raw[self.box_top_y - edge:self.box_bottom_y + edge,
                                self.box_left_x - edge:self.box_right_x + edge]

        _, subset_bin = cv2.threshold(subset, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # get the max id of segment so that the new segments are higher
        max_segment_id = np.amax(self.image_unsupervised)

        self.event_add_to_log("Cluster: Reclustering begins")

        if subset_bin is None:
            self.event_add_to_log("Cluster: Reclustering failed. Please try again")
            self.bool_image_locked = False
            return

        # the actual resegmenting
        try:
            subset_segmented = segment(subset_bin, nr_clusters, max_segment_id)
        except(Exception,):
            self.event_add_to_log("Cluster: Reclustering failed. Please try again")
            self.bool_image_locked = False
            return

        self.event_add_to_log("Cluster: Reclustering finished")

        # make the subset small again
        subset_segmented = subset_segmented[edge:subset_segmented.shape[0] - edge,
                                            edge:subset_segmented.shape[1] - edge]

        # make backup
        self.image_unsupervised_backup = copy.deepcopy(self.image_unsupervised)

        # make sure everything works
        if self.image_unsupervised[self.box_top_y:self.box_bottom_y,
           self.box_left_x:self.box_right_x].shape != subset_segmented.shape:
            self.event_add_to_log("Error: Something went wrong with resegmenting. PLease try again")
        else:
            # change
            self.image_unsupervised[self.box_top_y:self.box_bottom_y,
                                    self.box_left_x:self.box_right_x] = subset_segmented

        # reset boxes
        self.box_top_y = None
        self.box_bottom_y = None
        self.box_left_x = None
        self.box_right_x = None

        # remove patches
        [p.remove() for p in reversed(self.ax_unsupervised.patches)]

        # update plots
        self.event_update_plots()

    # let the user undo the last action
    def event_undo_last_action(self):

        self.event_add_to_log("Start undoing last change")

        # check if currently an action is done
        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        # check if image is loaded
        if self.bool_image_loaded is False:
            self.event_add_to_log("Error: no image is loaded.")
            self.bool_image_locked = False
            return

        if self.image_final_backup is None and self.image_unsupervised_backup is None:
            self.event_add_to_log("Error: no backup available")
            self.bool_image_locked = False
            return

        # if there's a backup for box drawing use this
        if self.image_unsupervised_backup is not None:
            self.image_unsupervised = copy.deepcopy(self.image_unsupervised_backup)
        else:
            # change the images
            self.image_final = copy.deepcopy(self.image_final_backup)

        self.image_unsupervised_backup = None
        self.image_final_backup = None

        self.event_update_plots()
        self.event_refresh_tree()

        self.event_add_to_log("Undoing finished")

        self.bool_image_locked = False

    # set the class to a certain cluster
    def event_set_class(self):

        # check if currently an action is done
        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        # check if image is loaded
        if self.bool_image_loaded is False:
            self.event_add_to_log("Error: no image is loaded.")
            self.bool_image_locked = False
            return

        # get x and y coordinates
        x = self.text_x_input.get("1.0", END)
        y = self.text_y_input.get("1.0", END)

        # check if it is a valid number
        try:
            x = int(x)
            y = int(y)
        except (Exception,):
            self.event_add_to_log("Error: X or Y is not a valid number")
            self.bool_image_locked = False
            return

        # check if numbers in range
        if x < 0 or x > self.image_unsupervised.shape[0]:
            self.event_add_to_log("Error: X is not a valid value")
            self.bool_image_locked = False
            return
        elif y < 0 or y > self.image_unsupervised.shape[1]:
            self.event_add_to_log("Error: Y is not a valid value")
            self.bool_image_locked = False
            return

        # check segmentId
        if self.combo_class.get() == '':
            self.event_add_to_log("Error: Please select a class")
            self.bool_image_locked = False
            return

        # get segment id
        segment_id = self.classes.index(self.combo_class.get())
        print(segment_id)

        # call the function to set the class to an image
        # the actual setting of the class
        img_output, img_clustered = sc.set_class(self.image_unsupervised,
                                                 self.image_final,
                                                 x, y,
                                                 segment_id,
                                                 self.image_clustered,
                                                 self.bool_scribble_in_image,
                                                 self.bool_box_in_image,
                                                 self.bool_overwrite_classes.get(),
                                                 )

        # save first a copy as a backup and then save the output
        self.image_final_backup = copy.deepcopy(self.image_final)
        self.image_final = copy.deepcopy(img_output)

        # save the preclustered so that the future class setting is faster
        self.image_clustered = copy.deepcopy(img_clustered)

        # remove the scribbles (if wanted)
        if self.bool_scribble_in_image and self.bool_remove_scribbles_after_change.get():
            self.image_unsupervised = copy.deepcopy(self.image_unsupervised_original)
            self.bool_scribble_in_image = False

        # update the image plots
        self.event_update_plots()

        # refresh the tree
        self.event_refresh_tree()

        # reset the x,y,class elements
        self.text_x_input.delete(1.0, "end")
        self.text_y_input.delete(1.0, "end")

        # reset the backup of re-segmentation
        self.image_unsupervised_backup = None

        # reset the lock
        self.bool_image_locked = False

        # write to log
        self.event_add_to_log("Class {} assigned to segment {}".format(self.combo_class.get(),
                                                                       self.image_unsupervised[y, x]))

    # set a certain class to all elements that don't have a class
    def event_set_class_all(self):

        # check if currently an action is done
        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        # check if image is loaded
        if self.bool_image_loaded is False:
            self.bool_image_locked = False
            self.event_add_to_log("Error: no image is loaded.")
            return

        # check segmentId
        if self.combo_class.get() == '':
            self.bool_image_locked = False
            self.event_add_to_log("Error: Please select a class")
            return

        # ask if this should really be done
        msg_box = messagebox.askquestion('Set all classes',
                                         'Are you sure you want to set all classes?', icon='warning')
        if msg_box == 'yes':
            # get segment id
            segment_id = self.classes.index(self.combo_class.get()) + 1

            # remove scribbles if available
            if self.bool_scribble_in_image:
                self.image_unsupervised = copy.deepcopy(self.image_unsupervised_original)
                self.bool_scribble_in_image = False

            # change the values
            self.image_final[self.image_final == 0] = segment_id

            # refresh the trees
            self.event_refresh_tree()

            # update the plots
            self.event_update_plots()

            # reset the state
            self.bool_image_locked = False

        else:
            self.bool_image_locked = False

    # remove all actions and load the original image again
    def event_reset_image(self):

        # check if currently an action is done
        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        # check if images is loaded
        if self.bool_image_loaded is False:
            self.event_add_to_log("Error: no image is loaded.")
            self.bool_image_locked = False
            return

        # ask if really want to reset
        msg_box = messagebox.askquestion('Reset image',
                                         'Are you sure you want to reset the image?', icon='warning')
        if msg_box == 'yes':
            # reset the images
            self.image_final = np.zeros_like(self.image_final)
            self.image_unsupervised = copy.deepcopy(self.image_unsupervised_original)

            # draw the images again
            self.event_update_plots()

            self.bool_image_locked = False
        else:
            self.bool_image_locked = False

    # show the selected image in the default image view
    def event_show_selected_image(self):

        if self.selected_file_id is None:
            print("This should not happen")
            return

        # get the image path
        image_path = self.params["path_images"] + "/" + self.selected_file_id + ".tif"

        # select the suitable viewer
        image_viewer = {'linux': 'xdg-open', "win32": 'explorer'}[sys.platform]

        # open the image
        spc.Popen([image_viewer, image_path])

    # save the segmented image to a file
    def event_save_image(self):

        # check if currently an action is done
        if self.bool_image_locked:
            self.event_add_to_log("Error: Currently another action is done. Please try again later.")
            return
        else:
            self.bool_image_locked = True

        # check if image is loaded
        if self.bool_image_loaded is False:
            self.event_add_to_log("Error: no image is loaded.")
            self.bool_image_locked = False
            return

        # last check that should usually never occur
        if self.selected_file_id is None:
            print("That should not happen")
            self.bool_image_locked = False
            return

        # replace not set with unknown
        self.image_final[self.image_final == 0] = 7
        self.image_final[self.image_final > 100] = 7

        img_output = cv2.resize(self.image_final, dsize=(self.img_orig_dims[1],
                                                         self.img_orig_dims[0]),
                                interpolation=cv2.INTER_NEAREST)

        # the pad function (to add the border again)
        def pad_with(vector, pad_width, _, kwargs):
            pad_value = kwargs.get('padder', 0)
            vector[:pad_width[0]] = pad_value
            if pad_width[1] != 0:  # <-- the only change (0 indicates no padding)
                vector[-pad_width[1]:] = pad_value

        top_add = self.border_bounds[0]
        left_add = self.border_bounds[2]
        bottom_add = self.img_orig_dims[0] - self.border_bounds[1]
        right_add = self.img_orig_dims[1] - self.border_bounds[3]

        # add border to the edge
        img_output = np.pad(img_output, ((top_add, bottom_add), (left_add, right_add)), pad_with, padder=7)

        # change datatype so that we can save memory
        img_output = img_output.astype('int')

        # finale security check
        if img_output.shape[0] != self.img_orig_dims[0] or \
                img_output.shape[1] != self.img_orig_dims[1]:
            self.event_add_to_log("Shapes are not correct. Something went wrong")
            return

        # save image
        cv2.imwrite(self.params["path_segmented"] + "/supervised/" + str(self.selected_file_id) + ".tif", img_output)

        # remove image to done folder
        os.rename(self.params["path_segmented"] + "/unsupervised" + str(self.selected_file_id) + ".tif",
                  self.params["path_segmented"] + "/unsupervised/done/" + str(self.selected_file_id) + ".tif")

        # remove image from files-list
        self.list_of_files.remove(self.selected_file_id)

        # add to log
        self.event_add_to_log("{} is saved".format(self.selected_file_id))

        self.bool_image_locked = False

    # this actions happens when hovering the mouse over the images, used for updating the text over the images but also
    # for the scribbling
    def event_mouse_hover(self, event):

        # mouse hover should not work when no image is loaded
        if self.bool_image_loaded is False:
            return

        # if outside the images, nothing should happen and no values should be displayed
        if event.xdata is None or event.ydata is None:
            self.string_var_x_pos.set("x: ...")
            self.string_var_y_pos.set("y: ...")
            self.string_var_unsupervised_class.set("class: ...")
            self.string_var_unsupervised_class.set("final class: ...")
            return

        # get x and y position of mouse
        x, y = int(event.xdata), int(event.ydata)

        #
        # update text
        #

        # get the final id
        final_id = int(self.image_final[y, x])
        if final_id == 0:
            final_class = "not set"
        else:
            final_class = self.classes[final_id]

        # set the values based on position
        self.string_var_x_pos.set("x: " + str(x))
        self.string_var_y_pos.set("y: " + str(y))
        self.string_var_unsupervised_class.set("class: " + str(self.image_unsupervised[y, x]))
        self.string_var_supervised_class.set("final class: " + final_class)

        #
        # scribble
        #

        if self.bool_scribble_mode and self.bool_mouse_left_click and self.scribble_type.get() == 1:

            min_x = x - int(self.params["scribble_size"] / 2)
            if min_x < 0:
                min_x = 0

            min_y = y - int(self.params["scribble_size"] / 2)
            if min_y < 0:
                min_y = 0

            max_x = x + int(self.params["scribble_size"] / 2)
            if max_x > self.image_unsupervised.shape[1]:
                max_x = self.image_unsupervised.shape[1]

            max_y = y + int(self.params["scribble_size"] / 2)
            if max_y > self.image_unsupervised.shape[0]:
                max_y = self.image_unsupervised.shape[0]

            self.image_unsupervised[min_y:max_y, min_x:max_x] = 100
            self.bool_scribble_in_image = True

    # this happens when the mouse is clicked
    def event_mouse_click(self, event):

        # if no image is loaded nothing should happen
        if self.bool_image_loaded is False:
            return

        # clicks outside the image are uninteresting
        if event.xdata is None or event.ydata is None:
            return

        # set that left mouse is clicked
        self.bool_mouse_left_click = True

        # get click coordinates
        x, y = int(event.xdata), int(event.ydata)

        # box drawing mode replaces all others types
        if self.bool_box_drawing_mode:

            if self.box_top_y is None and self.box_left_x is None:
                self.box_top_y = y
                self.box_left_x = x

                self.event_add_to_log("Draw Box: TopY:{} LeftX:{}".format(y, x))

            else:
                self.box_bottom_y = y
                self.box_right_x = x
                self.event_add_to_log("Draw Box: BottomY:{} RightX:{}".format(y, x))

                if self.box_top_y >= self.box_bottom_y or self.box_left_x >= self.box_right_x:
                    self.event_add_to_log("Draw Box: Something went wrong. Please Draw the box again")

                    self.box_bottom_y = None
                    self.box_right_x = None
                else:

                    height = self.box_bottom_y - self.box_top_y
                    width = self.box_right_x - self.box_left_x

                    rect = patches.Rectangle((self.box_left_x, self.box_top_y), width, height, linewidth=1,
                                             edgecolor='r', facecolor='none')
                    self.ax_unsupervised.add_patch(rect)
                    self.fig_unsupervised.canvas.draw_idle()

                    self.window.config(cursor="arrow")

                    self.bool_box_in_image = True

                    self.bool_box_drawing_mode = False
                    self.bool_image_locked = False

            return

        # replace the x and y values in the textboxes with the coordinates of the click
        self.text_x_input.delete(1.0, "end-1c")
        self.text_x_input.insert("end-1c", x)
        self.text_y_input.delete(1.0, "end-1c")
        self.text_y_input.insert("end-1c", y)

        # for scribble type 2
        if self.bool_scribble_mode and self.scribble_type.get() == 2:

            if self.scribble_x is None and self.scribble_y is None:
                self.scribble_x = x
                self.scribble_y = y
                self.event_add_to_log("Scribble set to ({},{})".format(x, y))
            else:

                # this array will contain the line
                temp_img = np.zeros_like(self.image_unsupervised)

                # get line
                rr, cc = line(self.scribble_y, self.scribble_x, y, x)
                temp_img[rr, cc] = 1

                # buffer
                kernel = np.ones((self.params["scribble_size"], self.params["scribble_size"]))
                temp_img = np.int64(convolve2d(temp_img, kernel, mode='same') > 0)

                # get line values
                self.image_unsupervised[temp_img == 1] = 100

                # save for additional clicks
                self.scribble_x = x
                self.scribble_y = y

                # image is scribbled
                self.bool_scribble_in_image = True

    # this happens when the mouse is released
    def event_mouse_release(self, _):

        # set that left mouse is not clicked
        self.bool_mouse_left_click = False

        # update images when scribble is drawn
        if self.bool_scribble_mode and self.bool_scribble_in_image:
            self.event_update_plots()

    # in this function text is added to the log
    def event_add_to_log(self, text):

        # get the current time
        current_time = datetime.now().strftime("%H:%M:%S")

        # add to the log
        if self.scrolled_text_log is not None:
            self.scrolled_text_log.configure(state='normal')
            self.scrolled_text_log.insert('end', current_time + " - " + text + "\n")
            self.scrolled_text_log.configure(state='disabled')
            self.scrolled_text_log.see("end")


if __name__ == "__main__":
    # create the class
    segmentator = Segmentator(segmentator_params)

    # start the main loop
    segmentator.display_window()
