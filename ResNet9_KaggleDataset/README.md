# ResNet-9 Plant Diseases Detection

We use the data from [kaggle](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset), or you can get the dataset via the BaiduCloud link below.

Or we can call this task: PDD

>   链接: https://pan.baidu.com/s/1AlsTYyr4x0Ry3YX5PQS_8g  密码: 230a

## Description of this dataset

This dataset is created using offline augmentation from the original dataset. The original PlantVillage Dataset can be found [here](https://github.com/spMohanty/PlantVillage-Dataset).This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure. A new directory containing 33 test images is created later for prediction purpose.

>   Note: This description is given in the dataset itself

## What should we do

We need to build a model, which can classify between healthy and diseased crop leaves and also if the crop have any disease, predict which disease is it.

## How to use it

### download the dataset

look at the beginning of this article

after download it, rename the dir: "dataset"

you will see a folder like this:

```
.
├── new plant diseases dataset(augmented)
│   └── New Plant Diseases Dataset(Augmented)
├── New Plant Diseases Dataset(Augmented)
│   └── New Plant Diseases Dataset(Augmented)
└── test
    └── test

6 directories, 0 files
```

now, the whole folder would look like this:

```
.
├── README.md
└── ResNet9_KaggleDataset
    ├── config.py
    ├── dataset
    │   ├── new plant diseases dataset(augmented)
    │   │   └── New Plant Diseases Dataset(Augmented)
    │   ├── New Plant Diseases Dataset(Augmented)
    │   │   └── New Plant Diseases Dataset(Augmented)
    │   └── test
    │       └── test
    ├── model.py
    ├── __pycache__
    │   ├── config.cpython-36.pyc
    │   └── model.cpython-36.pyc
    ├── README.md
    ├── requirements.txt
    └── train.py

9 directories, 8 files

```

then run the train.py

>   python train.py

OK

### torch to onnx

if you want to deploy this model, you should turn it to `.onnx` 

run torch2onnx.py

>   python torch2onnx.py

OK
