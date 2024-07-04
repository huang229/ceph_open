import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFile
import torch
from config import config as cfg
from data.load_train_data2015 import get_files

def read_label(data_file, label_root, label_name):
    data_name = os.path.split(data_file)[-1].replace(".bmp", "")
    Tcoords = []
    for i in range(len(label_name)):
        label_file = os.path.join(label_root, label_name[i]) + "/" + data_name + ".txt"
        coords = []
        with open(label_file, "r") as file_:

            for line in file_.readlines():
                line = line.strip().split(",")
                if 2 == len(line):
                    coords.append([int(line[0]), int(line[1])])
        coords = np.array(coords)
        Tcoords.append(coords)

    return Tcoords

def data_crop(img, label_coords):

    height, width, _ = img.shape
    offx = 120
    offy = 80

    minxy = np.min(label_coords, axis=0)
    maxxy = np.max(label_coords, axis=0)
    x1, y1 = int(max(minxy[0] - offx, 0)), int(max(minxy[1] - offy, 0))
    x2, y2 = int(min(maxxy[0] + offx, width)), int(min(maxxy[1] + offy, height))

    img_crop = img[y1: y2, x1: x2, :]
    lheight, lwidth, _ = img_crop.shape

    img_crop = cv2.resize(img_crop,(cfg.IMG_Width, cfg.IMG_Height), interpolation=cv2.INTER_LINEAR)

    label_coords = label_coords - np.array([x1, y1])
    scalex = 1.0 * cfg.IMG_Width / lwidth
    scaley = 1.0 * cfg.IMG_Height / lheight
    label_coords = label_coords*np.array([scalex, scaley])

    return img_crop, label_coords



if __name__ == "__main__":


    file_path = "./ISBI_data/TrainingData/"
    label_path = "./ISBI_data/AnnotationsByMD/"
    img_list = []
    get_files(file_path, img_list, ".bmp")
    label_name = ["400_junior", "400_senior"]
    save_path = "./ISBI_data/traindata/"
    for i in range(len(img_list)):

        name = os.path.split(img_list[i])[-1].replace(".bmp", "")
        img = cv2.imread(img_list[i])

        Tcoords = read_label(img_list[i], label_path, label_name)
        label_coords = (Tcoords[0] + Tcoords[1]) / 2.0


        # height, width, _ = img.shape
        # img_crop, label_coords = data_crop(img, label_coords)
        # height, width, _ = img_crop.shape

        # img_crop = np.ascontiguousarray(np.flip(img_crop, axis=1))
        # label_coords[:, 0] = width - label_coords[:, 0]
        for pi in range(label_coords.shape[0]):
            coord = label_coords[pi].astype(np.int32)
            # if pi==11:
            #     coord[1] =coord[1]-10
            # if pi == 12:
            #     coord[1] = coord[1] + 10
            img = cv2.drawMarker(img, (coord[0], coord[1]), (255, 0, 0), 0, thickness=2)
            # cv2.putText(img, str(pi+1), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 7)

        # ehIMg = cv2.equalizeHist(img[:, :, 0])
        # ehIMg = np.stack([ehIMg,ehIMg,ehIMg], axis=-1)
        # cv2.imwrite(save_path + "ceph.jpg", img)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)
        # cv2.namedWindow("ehIMg", cv2.WINDOW_NORMAL)
        # cv2.imshow("ehIMg", ehIMg)
        cv2.waitKey(0)

        # np.save(save_path + name + "m.npy", img_crop)
        # np.save(save_path + name + "l.npy", label_coords)

        print("preprocess data i = ", i)


