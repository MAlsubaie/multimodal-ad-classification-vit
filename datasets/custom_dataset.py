import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataframe, img_shape=(128, 128, 128)):
        self.df = dataframe
        self.image_paths = self.df['ADNI_path'].values
        self.labels = self.df['Group'].values

        # Binary classification mapping
        self.label_names = {'CN': 0, 'AD': 1, "MCI": 2}
        self.num_classes = len(self.label_names)
        self.labels_binary = self.df['Group'].map(self.label_names).values
        
        print(f"{len(self.image_paths)} Images found with classification.")
        print(f"Samples per class: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        
        self.img_shape = img_shape

    def __len__(self):
        return len(self.image_paths)

    def resize_image(self, image_np):
        """ Resize the numpy image using scipy.ndimage.zoom """
        from scipy.ndimage import zoom
        scale_factors = [n/o for n, o in zip(self.img_shape, image_np.shape)]
        resized_image_np = zoom(image_np, scale_factors, order=1)  # Linear interpolation
        return resized_image_np
    
    def process_image(self, idx):
        try:
            image_path = self.image_paths[idx]
            label = self.labels_binary[idx]

            image_tensor = torch.load(image_path)

            if image_tensor.shape!= (1, 128, 128,128):
                image_tensor  = image_tensor.unsqueeze(0)

            return image_tensor, label
        
        except Exception as e:
            print(f"Exception in processing image {image_path}: {e}")
            return None, None

    def __getitem__(self, idx):
        img, lbl = self.process_image(idx)
        return img, lbl
