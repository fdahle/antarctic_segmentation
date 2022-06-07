import copy

import connect_to_db as ctd

"""
cut_off_edge(input_img, img_id, cut_method, edge, extra_edge, return_edge_dims, db_type, catch, verbose):
This function cut offs the edge from images based on different methods. The edge are the black part of the images from
the TMA archive that do not contain any semantic information. Note that the original input images are not changed (deep
copy is applied before). The edges can be removed with a default value (cut_method "default", value based in 'edge') or
more progressive based on fid points (cut_method "database", img_id required and needs fid points in all four corners
of the image)

INPUT:
    input_img (np-array): The raw image from where the edges should be cut off
    img_id (String, None): The image id of the input_img. Required if the edges should be cut off based on fid points.
    cut_method (String, "default"): specifies the cut method, can be ["default", "database", "auto"]
    edge (int, 700): The edge used when cutting via 'default'.
    extra_edge (int, 0): Something you want to remove something extra on top of the calculated border
    return_edge_dims (Boolean, 'False'): if yes also the edges (what is cut off how much) are returned
    db_type (String, "PSQL"): From which db do we extract the edges
    catch (Boolean, True): If true and somethings is going wrong (for example no fid points),
        the operation will continue and not crash
    verbose (Boolean, False): If true, the status of the operations are printed
OUTPUT:
    img (np-array): The raw image with removed edges. If something went wrong and catch=True, 'None' will be returned
    bounds [list]: how much is removed from the images from each side: x_left, x_right, y_top, y_bottom
"""


def cut_off_edge(input_img, img_id=None, cut_method="auto", edge=700, extra_edge=0,
                 return_edge_dims=False, db_type="PSQL", catch=True, verbose=False):
    # deep copy to not change the original
    img = copy.deepcopy(input_img)

    # check prerequisites
    if cut_method == "database":
        assert img_id is not None, "to get the borders from the database, an image id is required"

    # check if correct method was chosen
    cut_methods = ["auto", "default", "database"]
    if cut_method not in cut_methods:
        print("The specified method is incorrect. Following methods are allowed:")
        print(cut_methods)
        exit()

    if verbose:
        if img_id is None:
            print("Cut off edge for image with following method: {}".format(cut_method))
        else:
            print("Cut off edge for {} with following method: {}".format(img_id, cut_method))

    # cut for default
    if cut_method == "default":

        # add extra edge to the default one
        edge = edge + extra_edge

        # crop the image
        try:
            img = img[edge:img.shape[0] - edge, edge:img.shape[1] - edge]

            # save how much was cropped
            x_left = edge
            x_right = edge
            y_top = edge
            y_bottom = edge

            bounds = [x_left, x_right, y_top, y_bottom]

        except (Exception,) as e:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise e


    elif cut_method == "database":

        # build sql string to select the borders
        sql_string = "SELECT " \
                     "fid_mark_1_x, fid_mark_1_y, " \
                     "fid_mark_2_x, fid_mark_2_y, " \
                     "fid_mark_3_x, fid_mark_3_y, " \
                     "fid_mark_4_x, fid_mark_4_y " \
                     "FROM images_properties " + \
                     "WHERE image_id='" + img_id + "'"

        # get table_data
        table_data = ctd.get_data_from_db(sql_string, db_type=db_type, catch=catch, verbose=verbose)

        if table_data is None:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise ValueError("Data from table is invalid")

        # check if there is any none value in the data
        bool_none_in_data = False
        for key in table_data:
            if table_data[key][0] is None:
                bool_none_in_data = True
                break

        # catch possible errors
        if table_data is None or bool_none_in_data:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise ValueError("Data from table is invalid")

        # get left
        if table_data["fid_mark_1_x"][0] >= table_data["fid_mark_3_x"][0]:
            left = table_data["fid_mark_1_x"][0]
        else:
            left = table_data["fid_mark_3_x"][0]

        # get top
        if table_data["fid_mark_2_y"][0] >= table_data["fid_mark_3_y"][0]:
            top = table_data["fid_mark_2_y"][0]
        else:
            top = table_data["fid_mark_3_y"][0]

        # get right
        if table_data["fid_mark_2_x"][0] <= table_data["fid_mark_4_x"][0]:
            right = table_data["fid_mark_2_x"][0]
        else:
            right = table_data["fid_mark_4_x"][0]

        # get bottom
        if table_data["fid_mark_1_y"][0] <= table_data["fid_mark_4_y"][0]:
            bottom = table_data["fid_mark_1_y"][0]
        else:
            bottom = table_data["fid_mark_4_y"][0]

        x_left = left + extra_edge
        x_right = right - extra_edge
        y_top = top + extra_edge
        y_bottom = bottom - extra_edge

        bounds = [x_left, x_right, y_top, y_bottom]

        try:
            img = img[y_top:y_bottom, x_left:x_right]
        except (Exception,) as e:
            if catch:
                if return_edge_dims:
                    return None, None
                else:
                    return None
            else:
                raise e

    elif cut_method == "auto":

        db_img, db_bounds = cut_off_edge(input_img, img_id=img_id, cut_method="database",
                                         extra_edge=extra_edge, db_type=db_type, return_edge_dims=True, verbose=False,
                                         catch=True)

        if db_img is None:

            # try out the default cut off
            img, bounds = cut_off_edge(input_img, img_id=img_id, cut_method="default",
                                       extra_edge=extra_edge, return_edge_dims=True, verbose=False, catch=False)
        else:
            img = db_img
            bounds = db_bounds

    else:
        print("That should not happen")
        img = None
        bounds = None

    # return all stuff
    if return_edge_dims:
        return img, bounds
    else:
        return img
