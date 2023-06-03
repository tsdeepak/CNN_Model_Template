import torch
import torch.nn as nn
from tqdm import tqdm
from torchsummary import summary
from model import Net
import matplotlib.pyplot as plt

class utils:

    def GetCorrectPredCount(Class,pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

    def train(Class,model, device, train_loader, optimizer,train_acc,train_losses):
        model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # Predict
            pred = model(data)

            # Calculate loss
            loss = F.nll_loss(pred, target)
            train_loss+=loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            
            correct += Class.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        train_acc.append(100*correct/processed)
        train_losses.append(train_loss/len(train_loader))

        return train_acc, train_losses

    def test(Class,model, device, test_loader,test_acc,test_losses):
        model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)

                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

                correct += Class.GetCorrectPredCount(output, target)


        test_loss /= len(test_loader.dataset)
        test_acc.append(100. * correct / len(test_loader.dataset))
        test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        return test_acc , test_losses
        
    def train_data_transformation(Class, transforms):
        return transforms.Compose([
                transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
                transforms.Resize((28, 28)),
                transforms.RandomRotation((-15., 15.), fill=0),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ])
    
    def test_data_transformation(Class, transforms):
        return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.1307,))
                ])
    
    def create_dataloaders(Class,test_data,train_data,**kwargs):
        test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
        return test_loader,train_loader
    
    def model_summary(Class,input_size,device):
        model = Net().to(device)
        summary(model, input_size)
    
    def print_data(Class,train_loader):
        batch_data, batch_label = next(iter(train_loader)) 

        fig = plt.figure()

        for i in range(12):
            plt.subplot(3,4,i+1)
            plt.tight_layout()
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
            plt.title(batch_label[i].item())
            plt.xticks([])
            plt.yticks([])
    
    def print_loss_accuracy(Class,train_losses,train_acc,test_losses,test_acc):
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_acc)
        axs[1, 1].set_title("Test Accuracy")
    
    def getDevice(Class):
        cuda = torch.cuda.is_available()
        print("CUDA Available?", cuda)
        device = torch.device("cuda" if cuda else "cpu")
        return device
    
    def load_MNSIT(Class,datasets,train_transforms,test_transforms):
        train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
        test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
        return train_data, test_data
    