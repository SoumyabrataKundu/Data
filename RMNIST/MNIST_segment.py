import torch
import torchvision
import torchvision.transforms as transforms


import sys
sys.path.append('../../Steerable/')
from Steerable.Segmentation.Segment import SegmentationDataset
from Steerable.datasets.hdf5 import HDF5Dataset

#####################################################################################################
######################################## MNIST Dataset ##############################################
##################################################################################################### 

class RandomRotation:
    def __call__(self, image, target):
        angle = torch.randint(0, 360, (1,)).item()
        image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        target = transforms.functional.rotate(target, angle, interpolation=transforms.InterpolationMode.NEAREST)
        
        return image, target


#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 


def main(data_path, rotate):
    kwargs = {
        'image_shape' : (1,56,56),
        'min_num_per_image' : 2,
        'max_num_per_image' : 4,
        'max_iou' : 0.2,
        'transforms' : RandomRotation() if rotate else None
    }

    train_dataset = torchvision.datasets.MNIST(data_path, train=True, transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(data_path, train=False, transform=torchvision.transforms.ToTensor())
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [55000, 5000])
    
    train_dataset = SegmentationDataset(train_dataset, n_samples=10000, **kwargs)
    val_dataset = SegmentationDataset(val_dataset, n_samples=2000, **kwargs)
    test_dataset = SegmentationDataset(test_dataset, n_samples=50000, **kwargs)
    
    datasets = {'train' : train_dataset, 'val' : val_dataset, 'test' : test_dataset}

    hdf5file = HDF5Dataset('MNIST_segment.hdf5')
    for mode in datasets:
        if datasets[mode] is not None:
            hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/')
    parser.add_argument("--rotate", type=bool, default=False)

    args = parser.parse_args()

    main(**args.__dict__)
