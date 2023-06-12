import json

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset


TRAIN_LABELS = 'data/train.json'
TEST_LABELS = 'data/test.json'
IMG_DIR = 'data/images'


def load_image(id, img_dir=IMG_DIR):
    return mpimg.imread(f'{img_dir}/punk{"%04d" % id}.png')


def write_image(img_data, filepath):
    return mpimg.imsave(filepath, img_data)


def make_punks_df(labels_file):
    with open(labels_file) as f:
        punks = json.loads(f.read())
    df = pd.DataFrame.from_dict(punks, orient='index')
    df['img'] = df.apply(lambda row: [load_image(int(row.name))], axis=1)
    return df


def gimme_one(df, filter, row_num):
    row = df.iloc[row_num]
    data = np.array(row['img'][0])
    labels = np.array(row[filter])
    return data, labels


def any_to_one(i):
    x = True
    if i['any']:
        x = False
    return int(x)


class PunksDataset(Dataset):
    def __init__(self, filter, img_dir=IMG_DIR, train=False):
        self.filter = filter
        self.img_dir = img_dir

        self.labels_path = TRAIN_LABELS if train else TEST_LABELS
        punks_df = make_punks_df(self.labels_path)
        self.X = np.array([row[0] for row in punks_df['img'].to_numpy()])
        self.Y = punks_df[filter].to_numpy()
        # punks_df['any'] = punks_df[filter].apply(np.any, axis=1)
        # punks_df['none'] = punks_df.apply(lambda x: any_to_one(x), axis=1)
        # self.Y = punks_df[filter + ['none']].to_numpy()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
        # return gimme_one(self.punks_df, self.filter, idx)
