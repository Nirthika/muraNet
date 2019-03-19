import os
import numpy as np
import torch.utils.data as data
from PIL import Image

def getLabelInt(s):
    if '0\n' in s : lbl = 0
    elif '1\n' in s: lbl = 1
    else:
        print(s)
        raise ValueError("Unknown Label")
    return lbl

def getEachData(fn, srcDir):
    lblarr = []
    classArr = []
    imgFnArr = []
    file = open(fn, 'r')
    for line in file:
        line = line.split(',')
        imgFnArr.append(os.path.join(srcDir, line[0]))
        lblarr.append(getLabelInt(line[2]))
        classArr.append(line[1])
    file.close()
    return imgFnArr, lblarr, classArr

def printStatistics(lbl_arr, args):
    print("Total : ", len(lbl_arr))
    unique_lbls = set(lbl_arr)
    count = []
    for lbl in unique_lbls:
        alist = []
        for i, v in enumerate(lbl_arr):
            if v==lbl: alist.append(i)
        count.append(len(alist))
        print(f"Number of images in class  {lbl} is  {len(alist)}")
    
    sv = sum(count)
    weights = []
    for i, val in enumerate(count):
        weights.append(sv/val)
    tot = sum(weights)    
    print("Weights :")
    for val in weights:
        print(np.round(val/tot, 3), end=',')
        args.weights.append(np.round(val/tot, 3))
    print('\n-----------------\n')
    
class getData(data.Dataset):
    def __init__(self, args, train=True, transform=None):
        img_dir = './data/'
        if train:            
            annot_fn = 'MURA-v1.1/train_ILC.csv'
        else:
            annot_fn = 'MURA-v1.1/valid_ILC.csv'
        annot_fn = os.path.join(img_dir, annot_fn)
        self.fnArr, self.lbls, self.classArr = getEachData(annot_fn, img_dir)

        self.transform = transform
        self.train = train # training set or validation set
        printStatistics(self.lbls, args)
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        img_fn = self.fnArr[index]
        target = self.lbls[index]
        class_type = self.classArr[index]

        imagedata = default_loader(img_fn)
        
        if self.transform is not None:
            imagedata = self.transform(imagedata)

        return imagedata, target, class_type
            
            
    def __len__(self):
        return len(self.fnArr)
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
        
'''
df = getData()
df = getData(train=False)
'''
