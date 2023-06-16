import json

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import Dataset


PUNK_LABELS = 'data/punks.json'
TRAIN_LABELS = 'data/train.json'
TEST_LABELS = 'data/test.json'
IMG_DIR = 'data/images'


ALL_FILTERS = [
    '3DGlasses', 'alien', 'ape', 'bandana', 'beanie', 'bigBeard', 'bigShades',
    'blackLipstick', 'blondeBob', 'blondeShort', 'blueEyeShadow', 'buckTeeth',
    'cap', 'capForward', 'chinstrap', 'choker', 'cigarette', 'classicShades',
    'clownEyesBlue', 'clownEyesGreen', 'clownHairGreen', 'clownNose',
    'cowboyHat', 'crazyHair', 'darkHair', 'doRag', 'earring', 'eyeMask',
    'eyePatch', 'fedora', 'female', 'frontBeard', 'frontBeardDark', 'frown',
    'frumpyHair', 'goat', 'goldChain', 'greenEyeShadow', 'halfShaved',
    'handlebars', 'headband', 'hoodie', 'hornedRimGlasses', 'hotLipstick',
    'knittedCap', 'luxuriousBeard', 'male', 'medicalMask', 'messyHair',
    'mohawk', 'mohawkDark', 'mohawkThin', 'mole', 'mustache', 'muttonchops',
    'nerdGlasses', 'normalBeard', 'normalBeardBlack', 'orangeSide',
    'peakSpike', 'pigtails', 'pilotHelmet', 'pinkWithHat', 'pipe', 'policeCap',
    'purpleEyeShadow', 'purpleHair', 'purpleLipstick', 'redMohawk',
    'regularShades', 'rosyCheeks', 'shadowBeard', 'shavedHead', 'silverChain',
    'smallShades', 'smile', 'spots', 'straightHair', 'straightHairBlonde',
    'straightHairDark', 'stringyHair', 'tassleHat', 'tiara', 'topHat',
    'vampireHair', 'vape', 'vr', 'weldingGoggles', 'wildBlonde', 'wildHair',
    'wildWhiteHair', 'zombie'
]


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


def df_split(df, test_size=0):
    df_size = len(df)
    df_indices = list(range(df_size))
    np.random.shuffle(df_indices)

    a_idx = df_indices[test_size:]
    b_idx = df_indices[:test_size]

    return a_idx, b_idx


class PunksDataset(Dataset):
    def __init__(
            self, filter, test_size=0, img_dir=IMG_DIR, labels_path=PUNK_LABELS
    ):
        self.filter = filter
        self.img_dir = img_dir
        self.labels_path = labels_path

        punks_df = make_punks_df(self.labels_path)
        self.train_idx, self.test_idx = df_split(punks_df, test_size)

        self.X = np.array([row[0] for row in punks_df['img'].to_numpy()])
        self.Y = punks_df[filter].to_numpy()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
