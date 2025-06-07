import torch
import sys
import h5py

sys.path.append('../../Steerable/')
from Steerable.utils import HDF5, HDF5Dataset

def main():
    data_file = h5py.File('data/MoNuSeg.hdf5', 'r')
    train_dataset, val_dataset = torch.utils.data.random_split(HDF5(data_file, 'train'), [30, 7])
    datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : HDF5(data_file, 'test')}

    hdf5file = HDF5Dataset('MoNuSeg.hdf5')
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return

if __name__ == '__main__':
    main()
