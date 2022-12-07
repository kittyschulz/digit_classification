import torch 
import torchvision

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2d_1 = nn.Conv2d(3, 16, 3)
        self.conv2d_2 = nn.Conv2d(16, 32, 3)
        self.conv2d_3 = nn.Conv2d(32, 64, 3)
        self.fc_1 = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 11)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv2d_1(x)))
        x = self.maxpool(F.relu(self.conv2d_2(x)))
        x = self.maxpool(F.relu(self.conv2d_3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x


def train(model, trainloader, valloader, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    loss_list = []
    val_loss_list = []
    accuracy_list = []
    val_accuracy_list = []

    for _ in range(epochs): 
        running_loss = 0.
        accuracy = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
                
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1) 
            accuracy += torch.sum(predicted == labels)
            running_loss += loss.item()
            
        loss_list.append(running_loss/i)
        accuracy_list.append(accuracy/(32*i))
        running_loss = 0.0
        accuracy = 0

        scheduler.step()

        model.eval()
        val_loss = 0.
        val_accuracy = 0
    
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1) 

            val_loss += loss.item()
            val_accuracy += torch.sum(predicted == labels)

        val_loss_list.append(val_loss/i)
        val_accuracy_list.append(val_accuracy/(32*i))

    return loss_list, val_loss_list, accuracy_list, val_accuracy_list


def load_model(architecture, train=False):
    """
    Loads a specified model.

    architecture (str): a string specifying which model to load
    train (bool): a boolean specifying whether a model should be set to train() or eval()
    """
    if architecture == 'cnn':
        model = CNN()
        model.load_state_dict(torch.load('weights/cnn.pth', map_location=torch.device('cpu')))
        if train:
            model.train()
        else:
            model.eval()
        return model
    elif architecture == 'vgg':
        model = torchvision.models.vgg16()
        model.classifier[6] = nn.Linear(in_features=4096, out_features=11)
        model.load_state_dict(torch.load('weights/vgg.pth', map_location=torch.device('cpu')))
        if train:
            model.train()
        else:
            model.eval()
        return model
    elif architecture == 'vgg+imagenet':
        model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(in_features=4096, out_features=11)
        model.load_state_dict(torch.load('weights/vgg_imagenet.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        raise Exception('Invalid model specified.')