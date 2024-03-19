from torch.utils.data import DataLoader
from src.dataLoaders.MnistSvhnText.MnistSvhnText import  get_data_set_svhn_mnist
from src.eval_metrics.Classifiers.MnistSvhnClassifiers import SVHN_Classifier_shie,MNIST_Classifier_shie

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
device = "cuda"

train,test = get_data_set_svhn_mnist(with_text= False)

train_loader = DataLoader(train, batch_size=256,
                          shuffle=True,
                          num_workers=8, drop_last=True)

test_loader = DataLoader(test, batch_size=256,
                          shuffle= True,
                          num_workers=8, drop_last=True)

epochs = 50

if __name__ =="__main__":
    classifier = SVHN_Classifier_shie()
    classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(epochs):  # loop over the dataset multiple times
        classifier.train()
        running_loss = 0.0
        total_iters = len(train_loader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, data in tqdm(enumerate(train_loader)):
            # get the inputs
            x, targets = data
            x, targets = x["svhn"].to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = classifier(x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (i + 1) % 1000 == 0:
                print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                running_loss = 0.0
        print('Finished Training, calculating test loss...')
        if epoch%5 ==0 and epoch!=0:
            classifier.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for i, data in tqdm( enumerate(test_loader) ) :
                    x, targets = data
                    x, targets = x["svhn"].to(device), targets.to(device)
                    outputs = classifier(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            print('The classifier correctly classified {} out of {} examples. Accuracy: '
                '{:.2f}%'.format(correct, total, correct / total * 100))

    torch.save(classifier.state_dict(), "data/data_mnistsvhntext/clf/svhn_classifier.pt")



    classifier = MNIST_Classifier_shie()
    classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    for epoch in range(epochs):  # loop over the dataset multiple times
        classifier.train()
        running_loss = 0.0
        total_iters = len(train_loader)
        print('\n====> Epoch: {:03d} '.format(epoch))
        for i, data in tqdm(enumerate(train_loader)):
            # get the inputs
            x, targets = data
            x, targets = x["mnist"].to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = classifier(x)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if (i + 1) % 1000 == 0:
                print('iteration {:04d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / 1000))
                running_loss = 0.0
        print('Finished Training, calculating test loss...')
        if epoch%5 ==0 and epoch!=0:
            classifier.eval()
            total = 0
            correct = 0
            with torch.no_grad():
                for i, data in tqdm( enumerate(test_loader) ) :
                    x, targets = data
                    x, targets = x["mnist"].to(device), targets.to(device)
                    outputs = classifier(x)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            print('The classifier correctly classified {} out of {} examples. Accuracy: '
                '{:.2f}%'.format(correct, total, correct / total * 100))

    torch.save(classifier.state_dict(), "data/data_mnistsvhntext/clf/MNIST_classifier.pt")
