import os
import scipy.io
import sys
from PIL import Image

import torch
import torchvision
import numpy as np

sys.path.append('../../Steerable/')
from Steerable.datasets.hdf5 import HDF5Dataset


class PNGToTensor():
    def __init__(self):
        self.to_tensor = torchvision.transforms.ToTensor()
    def __call__(self, png_file):
        return self.to_tensor(Image.open(png_file))

class MATToTensor:
    def __call__(self, mat_file_name):
        mat_file = scipy.io.loadmat(mat_file_name, simplify_cells=True)
        
        class_labels = mat_file['class_labels']
        class_labels = torch.from_numpy(class_labels) if isinstance(class_labels, np.ndarray) else torch.tensor([class_labels])
        class_labels = torch.cat([torch.tensor([0]), class_labels])
        instance_map = torch.from_numpy(mat_file['instance_map']).long()

        return class_labels[instance_map].long()


class ConSepPatched(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train', image_transform=PNGToTensor(), target_transform=MATToTensor()):
        self.data_path = data_path
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = target_transform
        
        self.image_files = [s for s in os.listdir(os.path.join(self.data_path,'tiles')) if s.startswith(f"{self.mode}")]
        self.target_files = [s for s in os.listdir(os.path.join(self.data_path,'labels')) if s.startswith(f"{self.mode}")]

        self.image_files.sort()
        self.target_files.sort()
        assert len(self.image_files) == len(self.target_files)
        
        
    def __getitem__(self, index):
        assert self.image_files[index][:-4] == self.target_files[index][:-4]
        
        image = os.path.join(self.data_path,'tiles', self.image_files[index])
        target = os.path.join(self.data_path,'labels', self.target_files[index])
            
        if self.image_transform:
            image = self.image_transform(image)
        if self.target_transform:
            target = self.target_transform(target)
            
        return image, target
    
    def __len__(self):
        return len(self.image_files)


class ConSep(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train', image_transform=PNGToTensor(), target_transform=MATToTensor()):
        self.data_path = data_path
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = target_transform
        
        self.image_files = os.listdir(os.path.join(self.data_path,'tiles'))
        self.target_files = os.listdir(os.path.join(self.data_path,'labels'))
        self.image_files.sort()
        self.target_files.sort()
        
    def __getitem__(self, index):
        image_patches = [s for s in self.image_files if s.startswith(f"{self.mode}_{index+1}_")]
        target_patches = [s for s in self.target_files if s.startswith(f"{self.mode}_{index+1}_")]
        
        if not image_patches or not target_patches:
            raise IndexError()

        self.image_files.sort()
        assert len(image_patches) == len(target_patches)
        image = torch.zeros(3,1000,1000)
        target = torch.zeros(1000,1000, dtype=torch.long)
        
        for image_patch, target_patch in zip(image_patches, target_patches):
            assert image_patch[:-4] == target_patch[:-4]
            
            _, _, dim1_lower, dim1_upper, dim2_lower, dim2_upper = target_patch[:-4].split('_')
            loc = (Ellipsis, slice(int(dim1_lower),int(dim1_upper)), slice(int(dim2_lower),int(dim2_upper)))
            
            image[loc] = self.image_transform(os.path.join(self.data_path,'tiles', image_patch))
            target[loc] = self.target_transform(os.path.join(self.data_path,'labels', target_patch))
            
        return image, target
    
    def __len__(self):
        index = 0
        while True:
            image_patches = [s for s in self.image_files if s.startswith(f"{self.mode}_{index+1}_")]
            if not image_patches:
                break
            index +=1 
        return index

#####################################################################################################
######################################## Main Function ##############################################
#####################################################################################################         

        
def main(data_path):
    hdf5file = HDF5Dataset('ConSep.hdf5')
    for mode in ['train', 'test']:
        hdf5file.create_hdf5_dataset(mode, ConSep(data_path, mode))

    return
    
if __name__== '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/')
    args = parser.parse_args()

    main(**args.__dict__)
