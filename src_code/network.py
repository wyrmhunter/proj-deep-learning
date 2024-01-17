import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1,stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1,stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*64*64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)



    def forward(self, x):
        # Première couche de convolution suivie de ReLU et de max pooling
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # Deuxième couche de convolution suivie de ReLU et de max pooling
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # Redimensionner l'activation pour l'entrée de la première
        # couche entièrement connectée
        x = x.view(-1, 32 * 64 * 64)
        # Trois couches entièrement connectées avec ReLU entre chacune
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    