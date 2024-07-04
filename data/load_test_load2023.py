
import os
import cv2
import numpy as np
import torch
from config import config as cfg
index_se = np.array([11, 5, 6, 16, 21, 3, 7, 4, 14, 15, 18, 22, 26, 25, 29, 28, 8, 2, 12]).astype(np.int32)
def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(file_dir, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)


class TestData():
    def __init__(self, file_data, label_root):
        self.file_data = file_data
        self.label_root = label_root

        self.img_list = []
        self.img_data = {}
        self.DResolution = {}



        file_path = "G:/ISBI2023/cephalogram_machine_mappings.csv"
        with open(file_path, "r") as file_:


            for line in file_.readlines():
                line = line.strip().split(",")
                self.DResolution[line[0]] = line[2]


        # label_list = []
        # for i in range(len(self.file_data)):
        #     img = np.load(self.file_data[i])
        #
        #     label_coords = np.load(self.file_data[i].replace("m.npy", "l.npy"))
        #     label_coords = label_coords[index_se-1]
        #     self.img_data[i] = [self.file_data[i], img.astype(np.uint8), label_coords]
        #     print("preprocess data i = ", i)
        #     label_list.append(label_coords)


    def __len__(self):

       return len(self.file_data)

    def __getitem__(self, item):

        img = np.load(self.file_data[item])

        label_coords = np.load(self.file_data[item].replace("m.npy", "l.npy"))
        label_coords = label_coords[index_se - 1]
        dataname = os.path.split(self.file_data[item])[-1].replace("m.npy", "")
        resov = np.array([float(self.DResolution[dataname])])


        minxy = np.min(label_coords, axis=0)
        img_crop, scalek, sizek = data_resize(img, minxy)

        img_crop = (img_crop / np.max(img_crop)).astype(np.float32)

        img_crop = torch.tensor(img_crop).permute(2, 0, 1)
        label_coords = torch.tensor(label_coords)
        scalek = torch.tensor(scalek)
        sizek = torch.tensor(sizek)

        return torch.tensor(img), img_crop, label_coords, scalek, sizek, torch.tensor(resov)



    def prepare_Data(self, file_root):

        get_files(file_root, self.img_list, ".bmp")

    def read_label(self, data_file):
        data_name = os.path.split(data_file)[-1].replace(".bmp", "")
        Tcoords = []
        for i in range(len(self.label_name)):
            label_file = os.path.join(self.label_root, self.label_name[i]) +"/" +data_name + ".txt"
            coords = []
            with open(label_file, "r") as file_:

                for line in file_.readlines():
                    line = line.strip().split(",")
                    if 2 == len(line):
                        coords.append([int(line[0]), int(line[1])])
            coords = np.array(coords)
            Tcoords.append(coords)

        return Tcoords

def data_resize(img, minxy):

    minxy = minxy//3
    offx = 0#np.random.randint(0, minxy[0], 1)[0]
    offy = 0#np.random.randint(0, minxy[1], 1)[0]

    img_crop = img[offy: , offx: , :]
    lheight, lwidth, _ = img_crop.shape
    ehIMg = cv2.equalizeHist(img_crop[:, :, 0])
    img_crop = np.stack([ehIMg, ehIMg, ehIMg], axis=-1)
    img_crop = cv2.resize(img_crop,(cfg.IMG_Width, cfg.IMG_Height), interpolation=cv2.INTER_LINEAR)

    scalex = 1.0 * lwidth / cfg.IMG_Width
    scaley = 1.0 * lheight / cfg.IMG_Height
    scale = np.array([scalex, scaley])

    return img_crop, scale, [lwidth, lheight]






















