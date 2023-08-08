import torch
import torch.nn as nn
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.state_dim  = 5220          # all the features
        self.mid_dim    = 2**10       # net dimension
        self.action_dim = 2           # output (sell/nothing/buy)

        # make a copy of the model in ActorPPO (activation function in forward function)
        self.dropout50 = nn.Dropout(0.5)

        # Original initial layers
        self.fc1          = nn.Linear(self.state_dim, self.mid_dim)
        self.relu1        = nn.ReLU()
        self.fc2          = nn.Linear(self.mid_dim, self.mid_dim)
        self.relu2        = nn.ReLU()

        # Added layers
        self.fc3          = nn.Linear(self.mid_dim, self.mid_dim)
        self.relu3        = nn.ReLU()
        self.fc4          = nn.Linear(self.mid_dim, self.mid_dim)
        self.relu4        = nn.ReLU()

        # Original residual layers
        self.fc_end1      = nn.Linear(self.mid_dim, self.mid_dim)
        self.relu_end1    = nn.ReLU()
        self.fc_end2      = nn.Linear(self.mid_dim, 512)
        self.relu_end2    = nn.ReLU()
        self.fc_end3      = nn.Linear(512, 128)
        self.relu_end3    = nn.ReLU()
        self.fc_end4      = nn.Linear(128, 64)
        self.relu_end4    = nn.ReLU()
        self.fc_end5      = nn.Linear(64, 16)
        self.hw           = nn.Hardswish()
        self.fc_out       = nn.Linear(16, self.action_dim)

    def forward(self, x):
        x = x.float()

        # Original initial layers
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        # Added layers
        x = self.dropout50(x)
        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        x = self.relu4(x)

        # Original residual layers
        x = self.dropout50(x)
        x = self.fc_end1(x)
        x = self.relu_end1(x)

        x = self.fc_end2(x)
        x = self.relu_end2(x)

        x = self.dropout50(x)
        x = self.fc_end3(x)
        x = self.relu_end3(x)

        x = self.fc_end4(x)
        x = self.relu_end4(x)

        x = self.fc_end5(x)
        x = self.hw(x)

        x = self.fc_out(x)
        x = torch.sigmoid(x)
        return x