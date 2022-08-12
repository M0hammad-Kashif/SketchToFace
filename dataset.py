from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


def img_arr(path, file):
    img = Image.open(os.path.join(path, file))
    img = np.array(img)
    return img


class CustomImageDataset(Dataset):
    def __init__(self, face_dir, sketch_dir, transform=None):
        self.face_dir = face_dir
        self.sketch_dir = sketch_dir
        self.face_files = os.listdir(self.face_dir)
        self.sketch_files = os.listdir(self.sketch_dir)
        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        face_file = self.face_files[idx]
        face_img = img_arr(self.face_dir, face_file)
        img_no = face_file.split('.')[0]
        sketch_img = img_arr(self.sketch_dir, img_no + '_edges.jpg')

        if self.transform:
            face_img = self.transform(image=face_img)['image']
            sketch_img = self.transform(image=sketch_img)['image']

        return sketch_img, face_img
