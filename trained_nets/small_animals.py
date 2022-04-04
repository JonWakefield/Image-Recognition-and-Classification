
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import ConcatDataset , Dataset
import ssl


ssl._create_default_https_context = ssl._create_unverified_context

# configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform data to help with training
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
TRANSFORM = transforms.Compose([ TPIL , transforms.RandomCrop(32 , padding=4, padding_mode='reflect'),
                                transforms.RandomHorizontalFlip(), TT ,NRM])
# Transforms object for testset with NO augmentation
TRANSFORM_NO_AUG = transforms.Compose([TT, NRM])

# training-parameters 
num_epochs = 10
batch_size = 4
learning_rate = 0.01


# CIFAR100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='trained_nets\\datasets\\', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='trained_nets\\datasets\\', train=False, download=True)

#classSmallDict = {'bear': 3 , 'fox': 34, 'lizard': 43, 'porcupine': 62, 'possum': 63, 'rabbit': 64, 'raccoon': 65, 'skunk': 74, 'snake': 77, 'squirrel': 79, 'wolf': 98}
#classSmall = {'bear': 3 , 'fox': 34, 'wolf': 98} cattle , lion , tiger

# print(train_dataset.classLabels)


classSmall = {'lizard': 43, 'porcupine': 62, 'rabbit': 64, 'raccoon': 65} #, 'snake': 77
classLabels = ('lizard', 'porcupine' , 'rabbit' , 'racoon' )


train_images = train_dataset.data
test_images = test_dataset.data
train_labels = train_dataset.targets
test_labels = test_dataset.targets


def get_class( x , y , i):
    
    # convert to numpy array
    y = np.array(y)

    # locate position of labels that equal to i
    pos_i = np.argwhere( y == i )

    # convert the result into a 1-D list
    pos_i = list(pos_i[: , 0])

    # collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i

# create custom dataset
class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc = TRANSFORM_NO_AUG):
        #datasets: a list of get_classSmall_i outputs, i.e. a list of list of images for selected classLabels
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        classSmall_label, index_wrt_classSmall = self.index_of_which_bin(self.lengths, i )
        img = self.datasets[classSmall_label][index_wrt_classSmall]
        img = self.transformFunc(img)
        return img , classSmall_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin( self , bin_sizes , absolute_index , verbose=False):
        
        #  Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        # which classSmall / bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("classSmall_label = ", bin_index)
        # which element of the fallent classSmall / bin does i correspond to?
        index_wrt_classSmall = absolute_index - np.insert(accum , 0 , 0)[bin_index]
        if verbose:
            print("index_wrt_classSmall = ", index_wrt_classSmall)
        
        return bin_index , index_wrt_classSmall
    

# choose which classLabels to import
train_set1 = DatasetMaker((get_class(train_images , train_labels, classSmall['lizard']), 
                            get_class(train_images, train_labels, classSmall['porcupine']),
                            # get_class(train_images, train_labels, classSmall['possum']),
                            get_class(train_images, train_labels, classSmall['rabbit']),
                            get_class(train_images, train_labels, classSmall['raccoon'])), TRANSFORM)
                            # get_class(train_images, train_labels, classSmall['skunk']),
                            # get_class(train_images, train_labels, classSmall['snake'])),)
                            # get_class(train_images, train_labels, classSmall['squirrel'])), 
                            
test_set1 = DatasetMaker([get_class(test_images , test_labels , classSmall['lizard']),
            get_class(test_images , test_labels, classSmall['porcupine']),
            # get_class(test_images , test_labels, classSmall['possum']),
            get_class(test_images , test_labels, classSmall['rabbit']),
            get_class(test_images , test_labels, classSmall['raccoon'])], TRANSFORM_NO_AUG)
            # get_class(test_images , test_labels, classSmall['skunk']),
            # get_class(test_images , test_labels, classSmall['snake'])], 
            # get_class(test_images, test_labels, classSmall['squirrel'])],
            




# create dataset loaders
train_loader = DataLoader( train_set1 , batch_size=batch_size , shuffle=True)
test_loader = DataLoader( test_set1 , batch_size=batch_size , shuffle = False)


training_its = 0

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
        print("entering ConvNet...")

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


model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# train_losses, test_losses = [], []

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    print("Entering training loop...")
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 50 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}') 

print('Finished Training')
PATH = 'detection\\trained_weights\\cnn_small_animals.pth'
torch.save(model.state_dict(), PATH)

# model.load_state_dict(torch.load(PATH))

# MAYBE USEFUL TO PLOT DATA
# plt.plot(train_losses, label='Training loss')
# plt.plot(test_losses, label='Validation loss')
# plt.legend(frameon=False)
# plt.show()


# model = ConvNet()
# model.load_state_dict(torch.load(PATH))
# model.eval()

# compare train vs test set to determine accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_classSmall_correct = [0 for i in range(10)]
    n_classSmall_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_classSmall_correct[label] += 1
            n_classSmall_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(4):
        acc = 100.0 * n_classSmall_correct[i] / n_classSmall_samples[i]
        print(f'Accuracy of {classLabels[i]}: {acc} %')

