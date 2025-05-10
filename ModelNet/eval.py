import torch
import numpy as np
import h5py
import os
import fnmatch
import sys

sys.path.append('../../Steerable/')
from Steerable.datasets.hdf5 import HDF5Dataset

class ModelNet10(torch.utils.data.Dataset):
    def __init__(self, data_path, size, mode):
        if mode not in ['train', 'test']:
            raise ValueError(f'Invalid mode {mode}. Should be one of train or test.')

        self.size = size
        self.orientations = 12
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if fnmatch.fnmatch(f, mode+'*.h5')]
        self.length = []

        for file in self.files:
            with h5py.File(file, 'r') as f:
                self.length.append(len(f['label']))

    def __getitem__(self, index):
        indices, label = self.get_indices(index // self.orientations)
        indices = self.rotate_point_cloud_3d_z(indices, (index%self.orientations) + 1)

        indices = ((indices + 1) * self.size/2).astype(int)
        image = torch.zeros(1, *[self.size]*3)
        for x_value, y_value, z_value in indices:
            image[0, x_value, y_value, z_value] = 1

        return image, torch.tensor(label)
    def get_indices(self, index):
        running_sum=0
        for file, length in zip(self.files, self.length):
            if index < running_sum + length:
                with h5py.File(file, 'r+') as f:
                    return f['data'][index - running_sum], f['label'][index - running_sum].item()
            running_sum += length

        raise IndexError('Index Out of Bounds.')

    def __len__(self):
        return sum(self.length)*self.orientations

    def rotate_point_cloud_3d_z(self, indices, orientation):
        angle = 2 * np.pi / (orientation)
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
        rotated_indices = np.dot(indices, R)

        return rotated_indices


def main(data_path, size):
    filename = 'ModelNet10_rotate_z' + str(size) + '_eval.hdf5'
    hdf5file = HDF5Dataset(filename)

    for mode in ['test']:
        dataset = ModelNet10(data_path=data_path, size=size, mode=mode)
        hdf5file.create_hdf5_dataset(mode, dataset)


if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/modelnet10/')
    parser.add_argument("--size", type=int, default=32)

    args = parser.parse_args()

    main(**args.__dict__)
