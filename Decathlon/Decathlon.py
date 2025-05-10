import torch
import os
import nibabel as nib
import json
import sys

sys.path.append('../../Steerable/')
from Steerable.datasets import HDF5Dataset

class Decathlon(torch.utils.data.Dataset):
    def __init__(self, data_path, image_transform = None, target_transform = None) -> None:

        self.image_transform = image_transform
        self.target_transform = target_transform
        
        with open(os.path.join(data_path, 'dataset.json'), 'r') as file:
            files = json.load(file)['training']
        self.image_files = [os.path.join(data_path, location['image']) for location in files]
        self.target_files = [os.path.join(data_path, location['label']) for location in files]
        
        self.n_samples = len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        target_file = self.target_files[index]
        assert os.path.basename(image_file) == os.path.basename(target_file)

        image = torch.from_numpy(nib.load(image_file).get_fdata())
        image = image.reshape(*image.shape[:3], -1).permute(3,0,1,2)
        target = torch.from_numpy(nib.load(target_file).get_fdata())
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return self.n_samples
    
    
def main(task):
    filename = task + '.hdf5'
    hdf5file = HDF5Dataset(filename)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(Decathlon(data_path= f'{task}/data/'), [0.7, 0.1, 0.2])
    datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}
 
    for mode in datasets:
        hdf5file.create_hdf5_dataset(mode, datasets[mode], variable_length=True)
    
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)

    args = parser.parse_args()

    main(**args.__dict__)
