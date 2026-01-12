import torch
from torch.utils.data import Dataset, DataLoader # pyright: ignore[reportMissingImports]
import torchvision.transforms.v2 as transforms # type: ignore
from pycocotools.coco import COCO
from torchvision.transforms.functional import crop
import os

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
from PIL import Image
import matplotlib.pyplot as plt


default_classify_transforms = transforms.Compose([
    transforms.Resize(size=(256,256)),
    transforms.RandomResizedCrop(
        size=(256,256), scale=(0.95, 1.0), ratio=(0.95,1.05)), 
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor()
        ])


# From each original image, 
# gets a crop of each of its bounding boxes, then applies transforms
class CocoSingleClassify(Dataset):
    def __init__(self, coco, img_root, transforms=default_classify_transforms):
        super().__init__()
        self.coco = coco
        self.img_root = img_root
        self.transforms = transforms
        # "category indices"
        self.cat_index_map = {}
        for i, id in enumerate(self.coco.getCatIds()):
            self.cat_index_map[id] = i 
        
        self.img_ids = []
        self.ann_ids = []

        _img_ids = coco.getImgIds()
        for img_id in _img_ids:
            img_dict = coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_root, img_dict["file_name"])
            
            # check if file exists
            if not os.path.exists(img_path):
                continue
            
            _ann_ids = coco.getAnnIds(imgIds=img_id)
            for _ann_id in _ann_ids:
                self.img_ids.append(img_id)
                self.ann_ids.append(_ann_id)
            
        assert len(self.img_ids) == len(self.ann_ids)
        

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, idx):
        ## Load image
        img_id = self.img_ids[idx]
        img_dict = self.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.img_root, img_dict["file_name"])
        # Force RGB for consistency
        img = Image.open(img_path).convert('RGB')
        # handle some entries w/ ndarrays
        if type(img) == np.ndarray:
            img = img.tolist()
        
        ## Load bbox and label
        ann = self.coco.loadAnns(self.ann_ids[idx])[0]
        
        # get bbox crop of image 
        x, y, w, h = ann['bbox']
        img = crop(img, y, x, h, w)
        # get label
        cat_id = ann['category_id']
        idx = self.cat_index_map[cat_id]   # "catagory index"
        target = torch.zeros(80)
        target[idx] = 1.
            
        
        return self.transforms(img), target



class CocoDataset(Dataset):
    def __init__(self, coco, img_root):
        super().__init__()
        self.coco = coco
        self.img_root = img_root
        self.transforms = transforms
        
        self.img_ids = []

        _img_ids = coco.getImgIds()
        if size is not None: # truncate
            _img_ids = _img_ids[:size]  

        for img_id in _img_ids:
            img_dict = coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_root, img_dict["file_name"])
            
            # check if file exists
            if not os.path.exists(img_path):
                continue

            self.img_ids.append(img_id)
            

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        ## Load image
        img_id = self.img_ids[idx]
        img_dict = self.coco.loadImgs(ids=[img_id])[0]
        img_path = os.path.join(self.img_root, img_dict["file_name"])
        # Force RGB for consistency
        img = Image.open(img_path).convert('RGB')
        # handle some entries w/ ndarrays
        if type(img) == np.ndarray:
            img = img.tolist()
            
        annIds = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annIds)
        target = torch.zeros(80)

        
        return self.transforms(img), img_id
