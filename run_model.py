import cv2
import torch
import numpy as np
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



cap = cv2.VideoCapture(0)
min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)
width, height = int(640/8),int(480/8)
net = Net(width, height)
PATH = "tuple_model_1.pth"
net.load_state_dict(torch.load(PATH))
while(True):
    img = cv2.flip(cap.read()[1], 1)
    image_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(image_YCrCb, min_YCrCb, max_YCrCb)
    img = cv2.bitwise_and(img, img, mask = skinRegionYCrCb)
    img_mod = cv2.resize(img, (width, height))
    img_mod = np.moveaxis(img_mod, -1, 0)
    output = net(torch.tensor([img_mod]))
    x, y = output[0]
    cv2.circle(img, (int(x.item()*640), int(y.item()*480)), 20, (255, 255, 0), -1)
    cv2.imshow("A", img)
    print(x.item(), y.item())
    if cv2.waitKey(2) & 0xFF == ord("w"):
        break