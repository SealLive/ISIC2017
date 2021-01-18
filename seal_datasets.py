import torch.utils.data as data
from seal_utils import readlines
from PIL import Image
import os, math
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def get_filename_list(gvs):
    imgs_dir_train = "/root/workspace/wlt_pytorch/dataset/ISIC2017/classification_data/ISIC-2017_Training_Data/"
    imgs_dir_test = "/root/workspace/wlt_pytorch/dataset/ISIC2017/classification_data/ISIC-2017_Test_v2_Data/"
    
    labels_dir = "/root/workspace/wlt_pytorch/dataset/"
    train_file_list = ["train_list_2017.txt"]
    test_file_list = ["test_list_2017.txt"]
    
    train_list = []
    for fn in train_file_list:
        lines = readlines(labels_dir+fn)
        for l in lines:
            path = imgs_dir_train+l.split("\t")[0]
            assert os.path.exists(path)
            class_label = str(int(l.split("\t")[1])//1)
            train_list.append(path+"\t"+class_label)
    
    test_list = []
    for fn in test_file_list:
        lines = readlines(labels_dir+fn)
        for l in lines:
            path = imgs_dir_test+l.split("\t")[0]
            assert os.path.exists(path)
            class_label = str(int(l.split("\t")[1])//1)
            test_list.append(path+"\t"+class_label)
            
    return train_list, test_list

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)
    
class MalignantFolder(data.Dataset):
    def __init__(self, filenames, transform=None, target_transform=None, loader=default_loader, extensions=IMG_EXTENSIONS):
        self.extensions = extensions
        
        pairs = self._find_pairs(filenames)
        self.samples = pairs
        if len(self.samples) == 0:
            raise(RuntimeError("Found 0 files in folders, \n" + 
                               "Supported extensions are: " + ",".join(extensions)))
            
        self.loader = loader

        self.transform = transform
        self.target_transform = target_transform

    def _find_pairs(self, filenames):
        pairs = []
        for line in filenames:
            image_path = line.split("\t")[0]
            tag = int(line.split("\t")[1])
            if os.path.exists(image_path) and has_file_allowed_extension(image_path, self.extensions):
                pairs.append((image_path, tag))
        return pairs
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        input_path, class_label = self.samples[index]
        input_image = self.loader(input_path)
        if self.transform is not None:
            input_image = self.transform(input_image)

        return input_image, class_label, input_path

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.x_root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str