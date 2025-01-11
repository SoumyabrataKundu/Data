import torch
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import tifffile as tiff
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import h5py


#####################################################################################################
#################################### Create dataset #################################################
##################################################################################################### 

def create_monuseg_dataset(datasets, name):
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



class AnnotationsToTensor:
    def __init__(self, target_shape):

        self.target_shape = target_shape

    def __call__(self, xml_file):
        
        polygons = self.parse_xml_to_polygons(xml_file)
        mask = self.create_mask_from_polygons(polygons, self.target_shape)
        
        return torch.from_numpy(mask)

    
    def parse_xml_to_polygons(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        polygons = []
        for region in root.findall(".//Region"):
            polygon = []
            for vertex in region.findall(".//Vertex"):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                polygon.append((x, y))
            polygons.append(polygon)
        
        return polygons

    def create_mask_from_polygons(self, polygons, image_size):
        mask = Image.new('L', image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        for polygon in polygons:
            draw.polygon(polygon, outline=1, fill=1)
        
        return np.array(mask)
    
#####################################################################################################
##################################### MoNuSeg Dataset ###############################################
##################################################################################################### 

class MoNuSeg(torch.utils.data.Dataset):
    def __init__(self, data_path, mode='train', image_transform = None, target_transform = None) -> None:

        self.image_transform = image_transform
        self.target_transform = target_transform
        
        self.data_path = data_path + '/PH2 Dataset images'
        image_paths = os.path.join(data_path, f'MoNuSeg{mode.capitalize()}Data', 'Tissue Images')
        target_paths = os.path.join(data_path, f'MoNuSeg{mode.capitalize()}Data', 'Annotations')
        
        self.image_files = [os.path.join(image_paths, image) for image in os.listdir(image_paths)]
        self.target_files = [os.path.join(target_paths, target) for target in os.listdir(target_paths)]           
        
        self.n_samples = len(self.image_files)

    def __getitem__(self, index):
        image = tiff.imread(self.image_files[index])
        target = self.target_files[index]
        
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return self.n_samples

def get_dataset(data_path) -> dict:
    image_shape = (1000, 1000)
    image_transform = transforms.ToTensor()
    target_transform = AnnotationsToTensor(image_shape)
    
    train_dataset = MoNuSeg(data_path, 'train', image_transform=image_transform, target_transform=target_transform)
    test_dataset = MoNuSeg(data_path, 'test', image_transform=image_transform, target_transform=target_transform)
    
    return {'train' : train_dataset, 'val': None, 'test' : test_dataset}


#####################################################################################################
######################################## Main Function ##############################################
##################################################################################################### 

def main():
    data_path = "./MoNuSeg/"
    name = os.path.join(data_path, "MoNuSeg.hdf5")
    target_shape = (1000, 1000)
    image_shape = (3,) + target_shape
    datasets = get_dataset(data_path)
    
    #if not os.path.isfile(name):
    create_hdf5_file(image_shape, target_shape, name)
    create_monuseg_dataset(datasets, name)
    
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