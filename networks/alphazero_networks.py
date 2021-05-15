import torch
from torch import nn

from networks.ffnn_networks import GenericFFNN


class GenericAlphaZeroNetwork(GenericFFNN):
    def forward(self, state):
        """
        Redefine the forward method of this network to be able to split the result into
        an probability distribution over the possible actions and an state-value estimator
        :param state: state to forward
        :return: state estimate and probability distribution over the possible actions
        """
        state = torch.Tensor(state)
        tmp = super().forward(state)

        # if its a multidimensional tensor of stacked stated, its more complex to split this
        if state.dim() == 2:
            return torch.stack([tensor[0] for tensor in tmp]), torch.stack(
                [torch.nn.functional.softmax(tensor[1:], dim=0) for tensor in tmp])
        return tmp[0], torch.nn.functional.softmax(tmp[1:], dim=0)


class TinyA0_Network(GenericAlphaZeroNetwork):
    def create_net(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_seqs + 1)
        )


class A0_Network(GenericAlphaZeroNetwork):
    def create_net(self):
        return nn.Sequential(
            nn.Linear(self.input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_seqs + 1)
        )
