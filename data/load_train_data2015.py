
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFile
import torch
from config import config as cfg

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_files(file_dir, file_list, type_str):

    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(file_dir, file_list, type_str)
        else:
            if file_.rfind(type_str) !=-1:
                file_list.append(path)



class TrainData():
    def __init__(self, file_root, label_root):

        self.file_root = file_root
        self.label_root = label_root

        self.img_list = []
        self.img_data = {}
        self.prepare_Data(self.file_root)
        self.label_name = ["400_junior", "400_senior"]
        label_list = []
        for i in range(len(self.img_list)):
            img = cv2.imread(self.img_list[i])

            Tcoords = self.read_label(self.img_list[i])
            label_coords = (Tcoords[0] + Tcoords[1]) / 2.0
            self.img_data[i] = [self.img_list[i], img.astype(np.uint8), label_coords]
            print("preprocess data i = ", i)
            label_list.append(label_coords)
        arr = np.mean(np.array(label_list), axis=0)



    def __len__(self):

       return len(self.img_list)

    def __getitem__(self, item):

        img = self.img_data[item][1]
        label_coords = self.img_data[item][2].copy()

        # if 0 == np.random.randint(0, 3, 1)[0]:
        #     scalev = np.random.randint(4, 8, 1)[0]
        #     label_coords[cfg.indx-1] = label_coords[cfg.indx-1] - cfg.DIFFv/(scalev*1.0)


        # tic = time.time()
        img_crop, label_coords_ = data_crop(img, label_coords)

        img_color = img_crop#randomColor(img_crop)

        img_rt, label_rt = randomRotation(img_color, label_coords_)

        img_rt = (img_rt / np.max(img_rt)).astype(np.float32)

        hot_map, offestxy, mask_ = genarater_hotmap(label_rt, cfg.IMG_Width, cfg.IMG_Height, sigma=10, sizek=30)
        hot_mapl,_, mk = genarater_hotmap(label_rt/32, cfg.IMG_Width//32, cfg.IMG_Height//32, sigma=10/16, sizek = 30//16)

        label_re = label_rt / np.array([[cfg.IMG_Height, cfg.IMG_Width]])

        img_rt = torch.tensor(img_rt).permute(2, 0, 1)
        hot_map = torch.tensor(hot_map)
        hot_mapl = torch.tensor(hot_mapl)
        offestxy = torch.tensor(offestxy)
        mask_ = torch.tensor(mask_)
        label_re= torch.tensor(label_re)

        return img_rt, hot_map, hot_mapl, offestxy, mask_, label_re



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



def data_crop(img, label_coords):

    height, width, _ = img.shape
    offv = 10
    minxy = np.min(label_coords, axis=0)
    maxxy = np.max(label_coords, axis=0)
    x1, y1 = np.random.randint(0, minxy[0]-offv, 1)[0], np.random.randint(0, minxy[1]-offv, 1)[0]

    x2, y2 = np.random.randint(min(maxxy[0]+offv, width)-1, max(maxxy[0]+offv, width), 1)[0], np.random.randint(min(maxxy[1]+offv, height)-1, max(maxxy[1]+offv, height),1)[0]


    img_crop = img[y1: y2, x1: x2, :]
    lheight, lwidth, _ = img_crop.shape

    img_crop = cv2.resize(img_crop,(cfg.IMG_Width, cfg.IMG_Height), interpolation=cv2.INTER_LINEAR)

    label_coords = label_coords - np.array([x1, y1])
    scalex = 1.0 * cfg.IMG_Width / lwidth
    scaley = 1.0 * cfg.IMG_Height / lheight
    label_coords = label_coords*np.array([scalex, scaley])



    return img_crop, label_coords
def randomColor(image):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    random_factor = np.random.randint(0, 17) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(8, 13) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(8, 16) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 16) / 10.  # 随机因子
    Sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    Sharp_image = np.asarray(Sharp_image)

    image = cv2.cvtColor(np.asarray(Sharp_image), cv2.COLOR_RGB2BGR)

    return image

def randomRotation(image, label_coords_, mode=Image.BILINEAR):
    """
     对图像进行随机任意角度(-20, 20度)旋转
    :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    :param image PIL的图像image
    :return: 旋转转之后的图像
    """
    image = Image.fromarray(image)
    random_angle = np.random.randint(-20, 20, 1)[0]

    img_rt = np.array(image.rotate(random_angle, mode))

    sinsta = np.sin(random_angle*np.pi/180)
    cossta = np.cos(random_angle*np.pi/180)

    cenp = np.array([cfg.IMG_Width/2, cfg.IMG_Height/2])
    label_cp = label_coords_ - cenp
    tmp = label_cp[:, 0].copy()
    label_cp[:, 0] = label_cp[:, 0] * cossta + label_cp[:, 1] * sinsta
    label_cp[:, 1] = -tmp * sinsta + label_cp[:, 1] * cossta
    label_rt = label_cp + cenp


    if np.max(label_rt[:, 0]) > (cfg.IMG_Width-2) or np.max(label_rt[:, 1]) > (cfg.IMG_Height-2):
        label_rt = label_coords_
        img_rt = image

    return img_rt, label_rt


def genarater_hotmap(label_, IMG_Width, IMG_Height, sigma=10, sizek = 30):


    sigma2 = 1/(sigma*sigma)
    X = np.arange(0, IMG_Width)
    Y = np.arange(0, IMG_Height)
    Y, X = np.meshgrid(Y, X, indexing='ij')
    hotmap = np.zeros((cfg.PointNms, IMG_Height, IMG_Width), np.float32)
    hotmapb = np.zeros((IMG_Height, IMG_Width), np.float32)
    offestxy = np.zeros((2, IMG_Height, IMG_Width), np.float32)
    mask_ = np.zeros((1, IMG_Height, IMG_Width), np.float32)
    offv = []
    for i in range(label_.shape[0]):
        corrd = label_[i]
        u1, u2 = min(int(round(corrd[0])), IMG_Width-1), min(int(round(corrd[1])),IMG_Height-1)
        x1, x2 = max(u1- sizek, 0), min(u1+ sizek+1, IMG_Width-1)
        y1, y2 = max(u2- sizek, 0), min(u2+ sizek+1, IMG_Height-1)
        X_ = X[y1: y2, x1: x2]
        Y_ = Y[y1: y2, x1: x2]

        XU1 = (X_ - u1) * (X_ - u1)
        YU2 = (Y_ - u2) * (Y_ - u2)
        Va = -(XU1+YU2) *sigma2
        gauv = np.exp(Va).astype(np.float32)

        hotmap[i, y1: y2, x1: x2] = np.maximum(hotmap[i,y1: y2, x1: x2], gauv)
        hotmapb[y1: y2, x1: x2] = np.maximum(hotmapb[y1: y2, x1: x2], gauv)
        offestxy[:, u2, u1] = np.array([corrd[0] - u1, corrd[1] - u2])
        mask_[0, u2, u1] = 1
        offv.append(np.array([corrd[0] - u1, corrd[1] - u2]))

    return hotmap, offestxy, mask_

















