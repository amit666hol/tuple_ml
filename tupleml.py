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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=self.width * self.height, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.width * self.height)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# classes = ['up', 'down', 'left', 'right', 'none', 'upright', 'upleft', 'downright', 'downleft']
#
#
# class DataSet(object):
#     def __init__(self, root, width, height):
#         self.root = root
#         # load all image files, sorting them to
#         # ensure that they are aligned
#         self.imgs = list(sorted(os.listdir(os.path.join("", self.root))))
#         self.width, self.height = width, height
#
#     def __getitem__(self, idx):
#         # load images
#         img_path = os.path.join("", self.root, self.imgs[idx])
#         img = cv2.imread(img_path)
#         try:
#             img = cv2.resize(img, (self.width, self.height))
#             img = np.moveaxis(img, -1, 0)
#         except Exception as e:
#             print(img_path)
#             raise e
#         img_label = img_path.split("\\")[1]
#         img_label = img_label.split(".")[0]
#         for i in range(len(img_label)):
#             if img_label[i].isdigit():
#                 img_label = img_label[:i]
#                 break
#         # img_label = ''.join([i for i in img_label if not i.isdigit()])
#         img_label = classes.index(img_label)
#         # print("Label: " + img_label + " Index: " + str(idx))
#         # there is only one class
#         return img, img_label
#
#     def __len__(self):
#         return len(self.imgs)


import torchvision.transforms as transforms
import torch.optim as optim


def main(root):
    # # get some random training images
    # # transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
    width, height = int(640/8), int(480/8)
    #
    # trainset = DataSet(root, width, height)
    #
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    #                                           shuffle=True, num_workers=2)
    net = Net(width, height)

    # VERY_NICE_criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    #
    # for epoch in range(5):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = VERY_NICE_criterion(outputs, torch.tensor(labels))
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         freq = 20
    #         if i % freq == freq-1:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss))
    #             running_loss = 0.0
    #
    # print('Finished Training')
    # PATH = "sans_model7c.pth"
    # torch.save(net.state_dict(), PATH)


if __name__ == '__main__':
    main("")
