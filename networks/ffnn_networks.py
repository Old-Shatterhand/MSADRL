import torch
from torch import nn

from networks.network import GenericNetwork


class GenericFFNN(GenericNetwork):
    def __init__(self, num_seqs, output_size=1):
        """
        Initialize the network
        :param num_seqs: number of sequences to align in the actual instance of the problem
        """
        super().__init__(num_seqs)
        self.output_size = output_size
        self.net = self.create_net()

    def __call__(self, output_size, *args, **kwargs):
        """
        set the output_size of the network after creation
        :param output_size: new output size
        """
        self.output_size = output_size

        # recreate the network
        self.net = self.create_net()

    def create_net(self):
        """
        Define a new network-architecture for the net in each child-class of this
        """
        pass

    def forward(self, state):
        """
        Forward the state through the network to find the action to select next
        :param state: state as linearized permutation of all sequences
        :return: index of sequence to align next
        """
        return self.net.forward(torch.Tensor(state))


class TinyValueFFNN(GenericFFNN):
    def create_net(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.output_size)
        )


class TinyPolicyFFNN(GenericFFNN):
    def create_net(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_seqs),
            nn.Softmax(dim=0)
        )


class ValueFFNN(GenericFFNN):
    def create_net(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Softmax(dim=0),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_size)
        )


class PolicyFFNN(GenericFFNN):
    def create_net(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Softmax(dim=0),
            nn.Linear(64, self.num_seqs),
            nn.Softmax(dim=0)
        )
