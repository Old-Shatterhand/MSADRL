import torch
from torch import nn

from networks.network import GenericNetwork


class GenericActorCriticNetwork(GenericNetwork):
    def __init__(self, num_seqs):
        """
        Initialize the network
        :param num_seqs: number of sequences to align in the actual instance of the problem
        """
        super().__init__(num_seqs)

    def forward(self, state):
        """
        Forward the state through the network to find the action to select next
        :param state: state as linearized permutation of all sequences
        :return: state_value for the state, action value for all actions
        """
        common = self.common_net.forward(torch.tensor(state).float())
        return self.value_net.forward(common), self.policy_net.forward(common)


class TinyACNetwork(GenericActorCriticNetwork):
    def __init__(self, num_seqs):
        # common beginning of actor and critic net
        super().__init__(num_seqs)
        self.common_net = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Softmax(dim=0),
        )

        # actor net
        self.policy_net = nn.Sequential(
            nn.Linear(16, num_seqs),
            nn.ReLU()
        )

        # critic net
        self.value_net = nn.Sequential(
            nn.Linear(16, 1)
        )


class AC_Network(GenericActorCriticNetwork):
    def __init__(self, num_seqs):
        super().__init__(num_seqs)
        self.common_net = nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Softmax(dim=0)
        )

        # actor net
        self.policy_net = nn.Sequential(
            nn.Linear(16, num_seqs),
            nn.ReLU()
        )

        # critic net
        self.value_net = nn.Sequential(
            nn.Linear(16, 1)
        )
