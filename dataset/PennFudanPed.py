import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class PennFudanDataset(Dataset):
    def __init__(self, root, split='train',transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))
        random_state=25
        test_size=0.33
        x_train,x_test,y_train,y_test=train_test_split(imgs,masks,test_size=test_size,random_state=random_state)
        if split=='train':
            self.imgs=x_train
            self.masks=y_train
        else:
            self.imgs=x_test
            self.masks=y_test
        self.labels=['background','people']

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax, 1])

        sample={}
        sample['img']=np.array(img).astype(np.float32)/255.0
        sample['annot']=np.array(boxes).astype(np.float32)
        return sample

    def __len__(self):
        return len(self.imgs)

    def image_aspect_ratio(self,idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        image = Image.open(img_path)
        return float(image.width) / float(image.height)