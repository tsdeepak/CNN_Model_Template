import torch
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt



def get_correct_predcount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


    
def train_data_transformation(transforms):
    return transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])

def test_data_transformation( transforms):
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.1307,))
            ])

def create_dataloaders(test_data,train_data,**kwargs):
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    return test_loader,train_loader



def print_data(train_loader):
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])

def print_loss_accuracy(train_losses,train_acc,test_losses,test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def getDevice():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    device = torch.device("cuda" if cuda else "cpu")
    return device

def load_MNSIT(datasets,train_transforms,test_transforms):
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    return train_data, test_data
    