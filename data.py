import os
import glob
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class img_dataset(Dataset):
    def __init__(self, path, debug=False):
        super(img_dataset, self).__init__()
        """
        path: a folder containing blurred/x and sharp/y folders
        """
        self.path = path

        #collect data points
        data = []
        x_paths = os.path.join(self.path, "blur_gamma", "*.png")
        xs = glob.glob(x_paths)
        for x in xs:
            basename = os.path.basename(x)
            y = os.path.join(self.path, "sharp", basename)
            data.append((x, y))
        if debug:
            self.data = data[:10]
        else:
            self.data = data

        self.convertor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x_tensor = self.convertor(Image.open(x))  #(3, 720, 1280)
        y_tensor = self.convertor(Image.open(y))  #(3, 720, 1280)
        return (x_tensor, y_tensor)
