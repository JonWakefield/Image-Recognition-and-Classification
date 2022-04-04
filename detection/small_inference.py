import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.utils import shuffle
import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import ConcatDataset , Dataset
from PIL import Image
from torch.autograd import Variable
import os.path


def small_inference():

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #, transforms.ToPILImage()
    # # configure device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    RC = transforms.RandomCrop(32, padding=4)
    RHF = transforms.RandomHorizontalFlip()
    RVF = transforms.RandomVerticalFlip()
    NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    TT = transforms.ToTensor()
    TPIL = transforms.ToPILImage()

    # Transforms object for trainset with augmentation
    transform_with_aug = transforms.Compose([TPIL, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    transform_no_aug = transforms.Compose([TT, NRM])

    Classes = ('lizard', 'porcupine' , 'rabbit' , 'racoon' )

    # CNN
    class ConvNet(nn.Module):
        def __init__(self):
            super(ConvNet, self).__init__()
            self.conv1 = nn.Conv2d( 3 , 32 , kernel_size=3 , padding=1)
            self.pool = nn.MaxPool2d( 2 , 2)
            self.conv2 = nn.Conv2d( 32 , 64 , kernel_size=3, padding=1)
            self.res1 = nn.Sequential( nn.Conv2d(64 , 64 , kernel_size=3, padding=1) , nn.Conv2d(64 , 64 , kernel_size=3, padding=1))
            self.pool = nn.MaxPool2d( 2 , 2)
            self.conv3 = nn.Conv2d( 64 , 128 , kernel_size=3 , padding=1)
            self.pool = nn.MaxPool2d( 2 , 2)
            self.conv4 = nn.Conv2d( 128 , 256 , kernel_size=3 , padding = 1)
            self.res2 = nn.Sequential( nn.Conv2d(256 , 256 , kernel_size=3, padding=1) , nn.Conv2d(256 , 256 , kernel_size=3, padding=1))
            self.pool = nn.MaxPool2d( 2 , 2)
            self.conv5 = nn.Conv2d( 256 , 512 , kernel_size=3, padding=1)
            self.res3 = nn.Sequential( nn.Conv2d(512 , 512 , kernel_size=3, padding=1) , nn.Conv2d(512 , 512 , kernel_size=3, padding=1))
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 4))

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.res1(x) + x
            x = self.pool(F.relu(self.conv3(x)))
            x = self.pool(F.relu(self.conv4(x)))
            x = self.res2(x) + x
            x = self.pool(F.relu(self.conv5(x)))
            x = self.res3(x) + x
            x = self.classifier(x)
            return x




    # Load trained-model
    model_PATH = 'detection\\trained_weights\\cnn_small_animals.pth'
    model = ConvNet().to(device)
    model.load_state_dict(torch.load(model_PATH))

    # load in image
    image = Image.open('detection\\images\\downsample_small_animal.png')

    img_tensor = transform_no_aug(image).to(device)

    with torch.no_grad():

        print("Predicting animal...")
        # tensor returns predictions, index with highest predictions % == predictions (add threshold)
        prediction = model(img_tensor.unsqueeze(0))
        predicted_class = prediction[0].argmax() # take prediction with highest value
        predicted = Classes[predicted_class]
        return predicted

