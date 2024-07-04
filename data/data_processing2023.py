

import os
import json
import filetype
import cv2
import numpy as np


def read_label(data_file, label_root, label_name):
    type_str = ["png", "jpg", "bmp", "jpeg"]
    if "png" in data_file:
        type = "png"
    if "jpg" in data_file:
        type = "jpg"
    if "bmp" in data_file:
        type = "bmp"
    if "jpeg" in data_file:
        type = "jpeg"

    data_name = os.path.split(data_file)[-1].replace("." + type, "")


    Tcoords = []
    for i in range(len(label_name)):
        label_file = os.path.join(label_root, label_name[i]) + "/" + data_name + ".json"
        coords = []
        names = []
        with open(label_file, "r") as jsf:
            label_dic = json.load(jsf)["landmarks"]
            for k, line_ in enumerate(label_dic):
                line = line_["value"]
                name = line_["title"]
                coordx, coordy = line["x"], line["y"]

                coords.append([int(coordx), int(coordy)])
                names.append(name)
        coords = np.array(coords)
        Tcoords.append(coords)

    return Tcoords, names

def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(file_dir, file_list, type_str)
        else:
            type = file_[-3:]
            if type in type_str:
                file_list.append(path)

if __name__ == "__main__":



    file_list = []
    type_str = ["png", "jpg", "bmp", "peg"]
    file_dir = "./ISBI2023/Train/Cephalograms/"
    label_path = "./ISBI2023/Train/Annotations/Cephalometric Landmarks/"
    get_files(file_dir, file_list, type_str)
    label_name = ["Junior Orthodontists", "Senior Orthodontists"]
    save_path = "./ISBI2023/train_data/"
    for i in range(0, len(file_list)):
        print(i)
        type_str = ["png", "jpg", "bmp", "jpeg"]
        if "png" in file_list[i]:
            type = "png"
        if "jpg" in file_list[i]:
            type = "jpg"
        if "bmp" in file_list[i]:
            type = "bmp"
        if "jpeg" in file_list[i]:
            type = "jpeg"

        name = os.path.split(file_list[i])[-1].replace("." + type, "")

        img = cv2.imread(file_list[i])
        print(name)
        Tcoords, names = read_label(file_list[i], label_path, label_name)
        label_coords = (Tcoords[0] + Tcoords[1]) / 2.0
        index_class = np.array([11, 5, 6, 16, 21, 3, 7, 4, 14, 15, 18, 22, 26, 25, 29, 28, 8, 2, 12]).astype(np.int32)
        label_coords = label_coords[index_class-1]
        names = np.array(names)[index_class-1]
        print(names)
        for pi in range(label_coords.shape[0]):
            coord = label_coords[pi].astype(np.int32)
            img = cv2.drawMarker(img, (coord[0], coord[1]), (0, 0, 255), 2, thickness=3)
            cv2.putText(img, str(pi+1)+"_"+names[pi], (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 7)
        #
        #
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        # np.save(save_path + name + "m.npy", img)
        # np.save(save_path + name + "l.npy", label_coords)


        print("")