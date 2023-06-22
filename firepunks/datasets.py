import json

import numpy as np
from torch.utils.data import Dataset

from . import images as IMGS


PUNK_LABELS = 'data/punks.json'
DF_IMG_COL = 'img'

ALL_LABELS = [
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


def split_labels(data, test_size=0):
    """
    Splits a list of data into two randomized sets of indexes.
    """
    data_indices = list(range(len(data)))
    np.random.shuffle(data_indices)

    a_idx = data_indices[test_size:]
    b_idx = data_indices[:test_size]

    return a_idx, b_idx


def filter_labels(all_labels, query):
    """
    Walks across a dict of labels data, eg how punks.json is stored, to create
    a list of a dicts that only contain the keys in list `query`.
    """
    return [{k: l.get(k) for k in query} for _, l in all_labels.items()]


class FirePunksDataset(Dataset):
    """
    Implements a pytorch Dataset for iterating across PIL image / label pairs.

    The images are converted from RGBA, as stored on disk, to RGB for ease of
    use with pytorch.
    """
    def __init__(
            self, labels, test_size=0, labels_path=PUNK_LABELS,
            img_dir=IMGS.IMG_DIR, transform=lambda x: x
    ):
        self.labels = labels
        self.labels_path = labels_path
        self.img_dir = img_dir
        self.test_size = test_size

        punks_data = load_labels(self.labels_path)
        split_idx = split_labels(punks_data, self.test_size)
        self.train_idx, self.test_idx = split_idx

        self.X = [transform(IMGS.load_image(idx))
                  for idx in range(len(punks_data))]

        self.Y = filter_labels(punks_data, labels)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def split_df(df, test_size=0):
    """
    Splits the row indexes of a dataframe into two randomized sets of indexes.

    This function is here for legacy purposes as part of converting cpunks to
    pytorch.
    """
    df_size = len(df)
    df_indices = list(range(df_size))
    np.random.shuffle(df_indices)

    a_idx = df_indices[test_size:]
    b_idx = df_indices[:test_size]

    return a_idx, b_idx


def load_labels_df(labels_path=PUNK_LABELS):
    """
    Loads the labels from punks.json and every image into a pandas dataframe.

    This function is here for legacy purposes as part of converting cpunks to
    pytorch.
    """

    import pandas as pd

    with open(labels_path) as f:
        punks = json.loads(f.read())
    df = pd.DataFrame.from_dict(punks, orient='index')

    def load_img(row):
        return [IMGS.load_mpimg(int(row.name))]

    df[DF_IMG_COL] = df.apply(load_img, axis=1)

    return df


class CPunksDataset(Dataset):
    """
    Implements a pytorch Dataset for iterating across the cpunks dataframe
    format.

    This function is here for legacy purposes as part of converting cpunks to
    pytorch.
    """
    def __init__(
            self, labels, test_size=0, img_dir=IMGS.IMG_DIR,
            labels_path=PUNK_LABELS
    ):
        self.labels = labels
        self.img_dir = img_dir
        self.labels_path = labels_path
        self.test_size = test_size

        punks_df = load_labels_df(self.labels_path)
        self.train_idx, self.test_idx = split_df(punks_df, self.test_size)

        self.X = np.array([row[0] for row in punks_df[DF_IMG_COL].to_numpy()])
        self.Y = punks_df[labels].to_numpy()

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
