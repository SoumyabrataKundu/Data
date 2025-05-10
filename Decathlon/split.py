import torch
import sys
import h5py

sys.path.append('../../Steerable/')
from Steerable.datasets.hdf5 import HDF5, HDF5Dataset

def main():
    data_file = h5py.File('data/Brain.hdf5', 'r')
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(HDF5(data_file, mode='train'), [340, 95, 49])
    datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}

    hdf5file = HDF5Dataset('BraTS.hdf5')
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return

if __name__ == '__main__':
    main()
