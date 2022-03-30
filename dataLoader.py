from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from skimage import io, transform
import pandas as pd

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # image, landmarks, targetMaps = sample['image'], sample['landmarks'], sample['targetMaps']
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        # print (h, w)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), mode='constant')
        landmarks = landmarks * [1 / (w - 1), 1 / (h - 1)]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]
        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, tlandmarks, targetMaps = sample['image'], sample['landmarks'], sample['targetMaps']
        image, tlandmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        landmarks = np.zeros(tlandmarks.shape)
        landmarks[:, 0], landmarks[:, 1] = tlandmarks[:, 1], tlandmarks[:, 0]
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}
        # 'targetMaps': [torch.from_numpy(targetMap).float() for targetMap in targetMaps]}


class LandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, landmarksNum=19):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.landmarkNum = landmarksNum

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:self.landmarkNum * 2 + 1].values.astype('float')
        landmarks = landmarks.reshape(-1, 2)

        sample = {'image': image, 'landmarks': landmarks}
        if self.transform:
            sample = self.transform(sample)

        return sample
