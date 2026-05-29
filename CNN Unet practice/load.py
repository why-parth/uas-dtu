import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch

class DroneDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, class_map, transform=None):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.class_map = class_map

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):

        img_path = os.path.join(
            self.image_dir,
            self.images[index]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[index]
        )

        
        image = Image.open(img_path).convert("RGB")

        mask = Image.open(mask_path).convert("RGB")

        
        image = image.resize((256,256))

        mask = mask.resize(
            (256,256),
            Image.NEAREST
        )

        image = np.array(image)

        mask = np.array(mask)

        label_mask = np.zeros(
            (mask.shape[0], mask.shape[1]),
            dtype=np.int64
        )

        for rgb, class_id in self.class_map.items():

            matches = np.all(
                mask == rgb, axis=-1
            )

            label_mask[matches] = class_id
        
        mask = label_mask

        image = torch.from_numpy(image)\
                    .permute(2,0,1)\
                    .float() / 255.0

        mask = torch.from_numpy(mask).long()

        if self.transform is not None:

            augs = self.transform(
                image=image,
                mask=mask
            )

            image = augs["image"]

            mask = augs["mask"]

        return image, mask


