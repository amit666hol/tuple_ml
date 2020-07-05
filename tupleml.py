import os
from PIL import Image
import numpy as np
import torch
import cv2
import torchvision

import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    channels = 3
    height = 480
    width = 640
    num_classes = 2

    def __init__(self, width, height):
        super(Net, self).__init__()
        self.width = width
        self.height = height
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=8960, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 8960)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


class DataSet(object):
    def __init__(self, root, width, height, resize_mul):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join("", self.root))))
        self.width, self.height = width, height
        self.resize_mul = resize_mul

    def __getitem__(self, idx):
        # load images
        img_path = os.path.join("", self.root, self.imgs[idx])
        img = cv2.imread(img_path)
        try:
            # Resize images to reduce model complexity and move axis so the model likes it
            img = cv2.resize(img, (int(self.width/self.resize_mul), int(self.height/self.resize_mul)))
            img = np.moveaxis(img, -1, 0)
        except Exception as e:
            # For debugging
            print(img_path)
            raise e
        # Get the x and y of the hand from the name
        img_name = img_path.split("\\")[1]
        img_name = img_name.split(".")[0]
        img_nums = img_name.split("_")
        # Convert the strings to integers
        img_nums = [int(num) for num in img_nums]
        # Map them from 0 to 1 cuz model big like
        img_tuple = (img_nums[0]/self.width, img_nums[1]/self.height)
        # there is only one class
        return img, img_tuple

    def __len__(self):
        return len(self.imgs)


import torchvision.transforms as transforms
import torch.optim as optim


def main(root):
    # get some random training images
    width, height, resize_mul = 640, 480, 8
    trainset = DataSet(root, width, height, resize_mul)

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)

    mseloss = nn.MSELoss()
    net = Net(int(width/resize_mul), int(height/resize_mul))

    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

    for epoch in range(5):  # loop over the dataset multiple times

        for i, data in enumerate(iter(trainset), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = net(torch.tensor([inputs]))[0]
            # print(output, target)
            # print(output.shape, torch.tensor(target).shape)
            loss = mseloss(output, torch.tensor(target))
            loss.backward()
            optimizer.step()

            # print statistics
            freq = 20
            if i % freq == freq-1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, loss.item()))
                print(output)
                print(target)

    print('Finished Training')
    PATH = "tuple_model_1.pth"
    torch.save(net.state_dict(), PATH)

def loss_test():
    width, height = 640, 480
    mseloss = nn.MSELoss()
    net = Net(width, height)
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    target = torch.tensor([0, 1]).float()
    for i in range(20):
        inputs = torch.rand(1, 3, width, height, requires_grad=True)
        optimizer.zero_grad()
        output = net(inputs)[0]
        print("output:", output, target)
        loss = mseloss(output, target)
        print("loss:", loss.item())
        loss.backward()
        optimizer.step()


def dataset_test():
    width, height = 640, 480
    dataset = DataSet("tuple_data", width, height, 8)
    a = next(iter(dataset))
    print(a[0].shape, a[1])


if __name__ == '__main__':
    main("tuple_data")
    # loss_test()
    # dataset_test()