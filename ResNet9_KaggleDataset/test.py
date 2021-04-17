from config import *
from model import *

import numpy as np
import torch
import torchvision
import onnxruntime
import time


def predict_image(img, model):
    xb = to_device(img.unsqueeze_(0), device="cpu")
    # yb = model(xb)
    yb = model.run([], {"input": xb.numpy()})
    # print(yb)
    yb = np.array(yb)
    pred = np.max(yb)

    return int(np.where(yb==pred)[-1])

test_dir = "./dataset/test"
test = ImageFolder(test_dir, transform=torchvision.transforms.ToTensor())
test_images = sorted(os.listdir(test_dir + '/test'))
model = onnxruntime.InferenceSession("./ResNet9_KaggleDataset.onnx")
train_dir, valid_dir = config()
train, _, _, _ = get_data(train_dir, valid_dir)

for i, (img, label) in enumerate(test):
    t0 = time.time()
    pred = predict_image(img, model)
    t1 = time.time()
    print('Label:', test_images[i], '\t', 'Predicted:', train.classes[pred], '\t', 'Time:', str(t1-t0))
