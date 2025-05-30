import torch
import h5py
import sys
sys.path.append('../../Steerable/')

from Steerable.Segmentation.Segment import SegmentationDataset
from Steerable.datasets.hdf5 import HDF5Dataset, HDF5
from Steerable.nn import rotate_3D_image

#####################################################################################################
######################################## MNIST Dataset ##############################################
##################################################################################################### 

class RandomRotation:
    def __call__(self, image, target):
        angle = torch.randint(0, 360, (3,))
        image = rotate_3D_image(image, angle, order=1)
        target = rotate_3D_image(target, angle, order=0)
        
        return image, target

def get_datasets(data_path, rotate=False) -> dict:
    kwargs = {
        'image_shape' : (1,64,64,64),
        'min_num_per_image' : 2,
        'max_num_per_image' : 4,
        'max_iou' : 0.2,
        'transforms' : RandomRotation() if rotate else None
    }

    hdf5file = h5py.File(data_path, 'r')
    train_dataset = HDF5(hdf5file, mode='train')
    test_dataset = HDF5(hdf5file, mode='test')
    
    train_dataset = SegmentationDataset(train_dataset, **kwargs)
    test_dataset = SegmentationDataset(test_dataset, **kwargs)
    
    return {'train' : train_dataset, 'val' : None, 'test' : test_dataset}


#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 


def main(data_path, rotate):
    datasets = get_datasets(data_path, rotate=rotate)
    hdf5file = HDF5Dataset('ModelNet10_segment.hdf5')
    for mode in datasets:
        if datasets[mode] is not None:
            hdf5file.create_hdf5_dataset(mode, datasets[mode])

    return
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--rotate", type=bool, default=False)

    args = parser.parse_args()

    main(**args.__dict__)
