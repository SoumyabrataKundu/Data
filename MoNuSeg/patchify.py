import torch
import h5py
import sys
sys.path.append('../../Steerable')

from Steerable.datasets.hdf5 import HDF5 
from Steerable.datasets.hdf5 import HDF5Dataset
from Steerable.Segmentation.Patchify import PatchifyDataset

def get_datasets() -> dict:
    train_dataset = HDF5(h5py.File('data/MoNuSeg.hdf5', 'r'), mode='train')
    test_dataset = HDF5(h5py.File('data/MoNuSeg.hdf5', 'r'), mode='test')

    return {'train' : train_dataset, 'val': None, 'test' : test_dataset}

def main():
    datasets = get_datasets()
    hdf5file = HDF5Dataset('MoNuSeg_segment250.hdf5')
    for mode in datasets:
        if datasets[mode] is not None:
            hdf5file.create_hdf5_dataset(mode, PatchifyDataset(datasets[mode], kernel_size=(250,250), stride=125))

    return

if __name__ == "__main__":
    main()
