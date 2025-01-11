import torch
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import h5py
from torchvision.transforms.functional import rotate

#####################################################################################################
#################################### Create dataset #################################################
##################################################################################################### 

def create_pascal_dataset(datasets, name):
    file = h5py.File(name, 'a')
    for mode in datasets:
        print(f"Mode : {mode} ...")
        dataset = datasets[mode]
        
        if datasets[mode] is not None:
            for index in range(len(dataset)):
                image, target = dataset[index]
                write_into_hdf5_file(file, mode, image, target)
                print(f"{index+1} / {len(dataset)}", end="\r")
        print('Done')
    file.close()


#####################################################################################################
#################################### Write into hdf5 File ###########################################
##################################################################################################### 

def create_hdf5_file(image_shape: tuple, 
                     target_shape: tuple,
                     name : str
                     ) -> None:

    with h5py.File(name, 'w') as f:
        for mode in ['train', 'test', 'val']:    
            f.create_dataset(mode + '_images', (0, ) + image_shape, maxshape=(None,) +  image_shape, chunks=True)
            f.create_dataset(mode + '_targets', (0,) + target_shape, maxshape=(None,) +  target_shape, chunks=True)
            
            
def write_into_hdf5_file(file,
                        mode : str,
                        image,
                        target, 
                        ) -> None:

    
    image_shape = tuple(image.size())
    target_shape = tuple(target.size())

    images = file[mode + '_images']
    targets = file[mode + '_targets']
    

    images.resize((len(images) + 1,) + image_shape)
    targets.resize((len(targets) + 1,) + target_shape)
    
    images[-1] = image
    targets[-1] = target
    
    return 
    
    
    
#####################################################################################################
##################################### PH2 Dataset Class #############################################
##################################################################################################### 

class Pascal(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train', rotate=False, image_transform = None, target_transform = None) -> None:
        self.image_transform = image_transform
        self.target_transform = target_transform
        file_path = os.path.join(data_path, 'ImageSets', mode+'.txt')
        self.rotate = rotate
        
        transforms.RandomRotation(degrees=(0, 360), interpolation=transforms.InterpolationMode.NEAREST)
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            self.image_files = [os.path.join(data_path + '/JPEGImages', line.strip() + '.jpg') for line in lines]
            self.target_files = [os.path.join(data_path + '/SegmentationClass', line.strip() + ".png") for line in lines]
            
        assert len(self.image_files) == len(self.target_files)
        self.n_samples = len(self.image_files)
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        target_file = self.target_files[index]
 
        assert os.path.basename(image_file)[:-4] == os.path.basename(target_file)[:-4]
        
        image = Image.open(image_file)
        target = Image.open(target_file)
        
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target) * 255
            target[target == 255] = 0
            target = target.type(torch.int)
        
        if self.rotate:
            degree = torch.randint(0, 360, (1,)).item()
            image = rotate(image, degree)
            target = rotate(target, degree)

        return image, target[0]
    
    def __len__(self):
        return self.n_samples
    
    
def get_dataset(data_path, rotate=False) -> dict:
    transformation = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        ])
    
    train_dataset = Pascal(data_path, mode='train', rotate=rotate, image_transform=transformation, target_transform=transformation)
    test_dataset = Pascal(data_path, mode='test', rotate=rotate, image_transform=transformation, target_transform=transformation)
    
    return {'train' : train_dataset, 'val': None, 'test' : test_dataset}
    



#####################################################################################################
###################################### Pascal Dataset ###############################################
##################################################################################################### 

def main(rotate=False):
    data_path = "./data"
    name = os.path.join(data_path, "RotPascal.hdf5" if rotate else "Pascal.hdf5")
    target_shape = (512, 512) 
    image_shape = (3,) + target_shape
    datasets = get_dataset(data_path, rotate=rotate)
    
    #if not os.path.isfile(name):
    create_hdf5_file(image_shape, target_shape, name)
    create_pascal_dataset(datasets, name)
    
    return


#####################################################################################################
######################################### View Dataset ##############################################
##################################################################################################### 


def view_dataset(file_name):
    modes = ['train', 'val', 'test']
    with h5py.File(file_name, mode='r') as f:
        for mode in modes:
            
            image_data = f[mode+'_images']
            target_data = f[mode+'_targets']
            assert len(image_data) == len(target_data)
            
            if not len(image_data) == 0:
            
                print(f'Shape of image in {mode} dataset : {image_data.shape}')
                print(f'Shape of target in {mode} dataset : {target_data.shape}')
                
                fig, ax = plt.subplots(1,2)
                fig.set_size_inches(10,5)
                index = torch.randint(0, len(target_data), (1,)).item()
                ax[0].imshow(torch.from_numpy(image_data[index]).permute(1,2,0))
                ax[1].imshow(target_data[index])
            
                plt.show()

    
    
        
        

                
    

if __name__ == '__main__':
    main(rotate=True)
