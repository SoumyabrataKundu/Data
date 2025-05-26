import torch
import sys
import h5py

sys.path.append('../../Steerable/')
from Steerable.utils import HDF5, HDF5Dataset

def main():
    data_file = h5py.File('data/PH2.hdf5', 'r')
    dataset = torch.utils.data.ConcatDataset([HDF5(data_file, 'train'), HDF5(data_file, 'test')])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [140, 20, 40])
    datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}

    hdf5file = HDF5Dataset('PH2.hdf5')
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return

if __name__ == '__main__':
    main()
