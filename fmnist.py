import os
from skimage import io
import torchvision as tv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def FMNIST(root, objective):
    character = [[] for i in range(10)]

    train_set = tv.datasets.FashionMNIST(root, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    val_set = tv.datasets.FashionMNIST(root, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    train_loader = DataLoader(dataset=train_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=2)
    val_loader = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=2)


    for X, Y in train_loader:  
        character[Y].append(X)
    for X, Y in val_loader:  
        character[Y].append(X)

    print(len(character[0]))

    meta_training = [[] for i in range(10)]
    meta_validation = [[] for i in range(10)]

    for idx, cls in enumerate(character):
        for i in range(0, 5000):
            meta_training[idx].append(cls[i])
        for i in range(5000, 6000):
            meta_validation[idx].append(cls[i])
        
    print(len(meta_training[0]))

    character = []

    os.mkdir(os.path.join(objective, 'train'))
    for i, per_class in enumerate(meta_training):
        character_path = os.path.join(objective, 'train', str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img = img[0]
            img = img.permute(1, 2, 0)
            img = img.numpy()
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)

    os.mkdir(os.path.join(objective, 'val'))
    for i, per_class in enumerate(meta_validation):
        character_path = os.path.join(objective, 'val', str(i))
        os.mkdir(character_path)
        for j, img in enumerate(per_class):
            img = img[0]
            img = img.permute(1, 2, 0)
            img = img.numpy()
            img_path = character_path + '/' + str(j) + ".jpg"
            io.imsave(img_path, img)


if __name__ == '__main__':
    root = './'
    objective = './data'
    FMNIST(root, objective)
    print("-----------------")