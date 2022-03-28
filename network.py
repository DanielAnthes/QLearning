import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    A simple fully connected network implementing the agents Policy function.
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(8,48)
        self.fc2 = nn.Linear(48,48)
        self.fc3 = nn.Linear(48,4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


