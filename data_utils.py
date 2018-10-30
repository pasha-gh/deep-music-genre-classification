import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

IMG_HEIGHT = 500
IMG_WIDTH = 500
NUM_CLASSES = 7

def load_data(img_dir):
    # naming convention: genre_musicID_slice.png
    # e.g. rock_music-0000_001.png
    assert os.path.isdir(img_dir)
    files = os.listdir(img_dir)
    features = _get_image_array(img_dir, files)
    labels = _get_label_array(files)
    assert features.shape[0] == labels.shape[0]
    return features, labels

def _get_image_array(img_dir, files):
    img_array = []
    for file in tqdm(files, ascii=True, ncols=80, desc='Reading images'):
        img = Image.open(img_dir + file)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)
        img = np.array(img)
        img = img[:,:,:3] # Use the RGB values
        img_array.append(img)
    return np.array(img_array)

def _get_label_array(files):
    label_dict = _get_label_dict(files)
    labels = []
    for file in files:
        label = file.split('_')[0]
        labels.append([label_dict[label]])
    one_hot = OneHotEncoder(n_values=NUM_CLASSES)
    labels = one_hot.fit_transform(labels).toarray()
    return np.array(labels)

def _get_label_dict(files):
    labels = set()
    for file in tqdm(files, ascii=True, ncols=80, desc='Reading labels'):
        label = file.split('_')[0]
        labels.add(label)
    label_dict = dict(enumerate(labels))
    label_dict = {v: k for k, v in label_dict.items()}
    return label_dict

def load_genre_specifit_data(img_dir):
    features, labels = load_data(img_dir)
    label_set = set()
    for label in labels:
        label_set.add(str(label))
    features_by_genre_dict = {}
    for label in label_set:
        features_by_genre_dict[label] = []
        for f, l in zip(features, labels):
            # print(str(l))
            if str(l) == label:
                # print('OMG')
                features_by_genre_dict[label].append(f)
        features_by_genre_dict[label] = np.array(features_by_genre_dict[label])
    return features_by_genre_dict
