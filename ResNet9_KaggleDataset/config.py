import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
    	return len(self.dl)


def config():
	# use your own dir
	data_dir = "/content/plant_diseases_detection/ResNet9_KaggleDataset/dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"
	train_dir = data_dir + "/train"
	valid_dir = data_dir + "/valid"
	diseases = os.listdir(train_dir)

	# you can see the diseases
	print("Total disease classes are: {}".format(len(diseases)))
	print(diseases)

	# get the kinds of plants
	plants = []
	number_of_diseases = 0
	for plant in diseases:
	    if plant.split('___')[0] not in plants:
	        plants.append(plant.split('___')[0])
	    if plant.split('___')[1] != 'healthy':
	        number_of_diseases += 1
	print(f"Unique Plants are: \n{plants}")
	print("Number of plants: {}".format(len(plants)))
	print("Number of diseases: {}".format(number_of_diseases))

	# get the numbers of all kinds of images
	nums = {}
	for disease in diseases:
		nums[disease] = len(os.listdir(train_dir + '/' + disease))
	# converting the nums dictionary to pandas dataframe passing index as plant name and number of images as column
	img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["numbers of images"])
	print(img_per_class)

	# get the num of images 
	n_train = 0
	for value in nums.values():
		n_train += value
	print(f"There are {n_train} images for training")

	return train_dir, valid_dir


def get_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	else:
		return torch.device("cpu")


def to_device(data, device):
	if isinstance(data, (list, tuple)):
		return [to_device(x, device) for x in data]
	return data.to(device, non_blocking=True)


def get_data(train_dir, valid_dir, device=get_device()):
	# you can modify these parameters
	torch.manual_seed(7)
	batch_size = 32

	train = ImageFolder(train_dir, transform=transforms.ToTensor())
	valid = ImageFolder(valid_dir, transform=transforms.ToTensor())
	img, label = train[0]
	print(img.shape, label)

	train_data = dataloader.DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
	valid_data = dataloader.DataLoader(valid, batch_size, num_workers=2, pin_memory=True)

	train_data = DeviceDataLoader(train_data, device)
	valid_data = DeviceDataLoader(valid_data, device)

	return train, valid, train_data, valid_data


def train_config():
	epochs = 5
	max_lr = 0.01
	grad_clip = 0.1
	weight_decay = 2e-4
	opt_func = torch.optim.Adam
	return epochs, max_lr, grad_clip, weight_decay, opt_func


if __name__ == '__main__':
	train_dir, valid_dir = config()
	_, _, _, _ = get_data(train_dir, valid_dir)
