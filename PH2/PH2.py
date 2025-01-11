import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import h5py
import matplotlib.pyplot as plt


#####################################################################################################
#################################### Create dataset #################################################
##################################################################################################### 

def create_ph2_dataset(datasets, name):
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


def get_max_size(data_path):
    data_path = data_path + '/PH2 Dataset images'
    image_folders = os.listdir(data_path)
    transform = transforms.ToTensor()            
    size = []
    for folder in image_folders:
        image = Image.open(os.path.join(data_path, folder, folder+'_Dermoscopic_Image', folder+'.bmp'))
        target = Image.open(os.path.join(data_path, folder, folder+'_lesion', folder+'_lesion.bmp'))

        image_tensor = transform(image)
        target_tensor = transform(target)
        assert image_tensor.shape[1:] == target_tensor.shape[1:]
        size.append(list(image_tensor.shape))
        
    size_tensor = torch.tensor(size)
    return torch.max(size_tensor[:,1]).item(), torch.max(size_tensor[:,2]).item()


class PadToShape:
    def __init__(self, target_shape, mode='constant', value=0):

        self.target_shape = target_shape
        self.mode = mode
        self.value = value

    def __call__(self, img):
        _, orig_h, orig_w = img.shape
        target_h, target_w = self.target_shape
        
        # Calculate the required padding
        pad_h = max(target_h - orig_h, 0)
        pad_w = max(target_w - orig_w, 0)
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

        # Apply padding
        padded_img = F.pad(img, padding, mode=self.mode, value=self.value)
        
        return padded_img


class PH2(torch.utils.data.Dataset):
    def __init__(self, data_path, image_transform = None, target_transform = None) -> None:

        self.image_transform = image_transform
        self.target_transform = target_transform
        
        self.data_path = data_path + '/PH2 Dataset images'
        self.image_folders = os.listdir(self.data_path)
        self.transform = transforms.ToTensor()            
        
        self.n_samples = len(self.image_folders)

    def __getitem__(self, index):
        folder = self.image_folders[index]
        image = Image.open(os.path.join(self.data_path, folder, folder+'_Dermoscopic_Image', folder+'.bmp'))
        target = Image.open(os.path.join(self.data_path, folder, folder+'_lesion', folder+'_lesion.bmp'))
        
        
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target[0]

    def __len__(self):
        return self.n_samples

def get_dataset(data_path) -> dict:
    max_size = (578, 770) #get_max_size(data_path)
    transformation = transforms.Compose([
            transforms.ToTensor(),
            PadToShape(max_size)
            ])
    
    dataset = PH2(data_path, image_transform=transformation, target_transform=transformation)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.75, 0.25])
    
    return {'train' : train_dataset, 'val': None, 'test' : test_dataset}


#####################################################################################################
######################################### PH2 Dataset ###############################################
##################################################################################################### 

def main():
    data_path = "../data/PH2/"
    name = os.path.join(data_path, "PH2.hdf5")
    target_shape = (578, 770) #get_max_size(data_path)
    image_shape = (3,) + target_shape
    datasets = get_dataset(data_path)
    
    #if not os.path.isfile(name):
    create_hdf5_file(image_shape, target_shape, name)
    create_ph2_dataset(datasets, name)
    
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
                fig.set_size_inches(5,10)
                index = torch.randint(0, len(target_data), (1,)).item()
                ax[0].imshow(torch.from_numpy(image_data[index]).permute(1,2,0))
                ax[1].imshow(target_data[index])
            
                plt.show()

    
    
        
        

                
    

