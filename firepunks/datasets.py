import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from . import images as IMGS


PUNK_LABELS = 'data/punks.json'
DF_IMG_COL = 'img'

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


def split_df(df, test_size=0):
    df_size = len(df)
    df_indices = list(range(df_size))
    np.random.shuffle(df_indices)

    a_idx = df_indices[test_size:]
    b_idx = df_indices[:test_size]

    return a_idx, b_idx


def load_labels_df(self, labels_path):
    with open(labels_path) as f:
        punks = json.loads(f.read())
    df = pd.DataFrame.from_dict(punks, orient='index')

    def load_img(row):
        return [IMGS.load_mpimg(int(row.name))]

    df[DF_IMG_COL] = df.apply(load_img, axis=1)

    return df


class CPunksDataset(Dataset):
    def __init__(
            self, labels, test_size=0, img_dir=IMGS.IMG_DIR,
            labels_path=PUNK_LABELS
    ):
        self.labels = labels
        self.img_dir = img_dir
        self.labels_path = labels_path

        punks_df = load_labels_df(self.labels_path)
        self.train_idx, self.test_idx = split_df(punks_df, test_size)

        self.X = np.array([row[0] for row in punks_df[DF_IMG_COL].to_numpy()])
        self.Y = punks_df[labels].to_numpy()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
