import glob
import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import zipfile
from pathlib import Path
import numpy as np
import shutil

classes = ['with_mask', 'without_mask', 'mask_weared_incorrect']
ROOT_DIR = "datasets"
DATA_FILE = "archive.zip"
IMAGE_EXT = ".png"

DATA_DIR = join(ROOT_DIR, "mask")
IMAGE_DIR = join(DATA_DIR, "images")
LABEL_DIR = join(DATA_DIR, "annotations")
PROCESSED_LABEL_DIR = join(DATA_DIR, "processed_annotations")
TRAIN_DATA_DIR = join(DATA_DIR, "train")
VALID_DATA_DIR = join(DATA_DIR, "valid")

with zipfile.ZipFile(join(ROOT_DIR, DATA_FILE), 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

Path(TRAIN_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(VALID_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(TRAIN_DATA_DIR + "/images").mkdir(parents=True, exist_ok=True)
Path(TRAIN_DATA_DIR + "/labels").mkdir(parents=True, exist_ok=True)
Path(VALID_DATA_DIR + "/images").mkdir(parents=True, exist_ok=True)
Path(VALID_DATA_DIR + "/labels").mkdir(parents=True, exist_ok=True)
Path(PROCESSED_LABEL_DIR).mkdir(parents=True, exist_ok=True)

# referred to https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x,y,w,h)

# referred to https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
def convert_annotation(classes, input_path, output_path):
    basename = os.path.basename(input_path)
    basename_no_ext = os.path.splitext(basename)[0]

    in_file = open(input_path)
    out_file = open(output_path + "/" + basename_no_ext + '.txt', 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

paths = glob.glob(LABEL_DIR + '/*.xml')

for xml_path in tqdm(paths):
    convert_annotation(classes, xml_path, PROCESSED_LABEL_DIR)

label_files = glob.glob(PROCESSED_LABEL_DIR + '/*.txt')

train_indices, valid_indices = train_test_split(
    np.arange(len(label_files)), test_size=0.2, random_state=42, shuffle=True)

train_labels = []
for idx in train_indices:
    train_labels.append(label_files[idx])

valid_labels = []
for idx in valid_indices:
    valid_labels.append(label_files[idx])

for label_path in train_labels:
    basename = os.path.basename(label_path)
    basename_no_ext = os.path.splitext(basename)[0]

    shutil.move(label_path, join(TRAIN_DATA_DIR, "labels", basename_no_ext + ".txt"))
    shutil.move(join(IMAGE_DIR, basename_no_ext + IMAGE_EXT), join(TRAIN_DATA_DIR, "images", basename_no_ext + IMAGE_EXT))

for label_path in valid_labels:
    basename = os.path.basename(label_path)
    basename_no_ext = os.path.splitext(basename)[0]

    shutil.move(label_path, join(VALID_DATA_DIR, "labels", basename_no_ext + ".txt"))
    shutil.move(join(IMAGE_DIR, basename_no_ext + IMAGE_EXT), join(VALID_DATA_DIR, "images", basename_no_ext + IMAGE_EXT))








