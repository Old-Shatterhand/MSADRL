import torch

from agents.network_solver import NetworkSolver
from networks.ffnn_networks import GenericFFNN
from utils.utils import linearize_state


class ValueAgent(NetworkSolver):
    def __init__(self, sequences, network_object: GenericFFNN, refinement=False, network_path="networks/FFNN.txt"):
        """
        Initialize the agent to train and select the alignment steps using an simple DQN-algorithm
        :param sequences: sequences to align
        :param network_object: network-object to use for training and function approximation in the algorithm
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param network_path: filepath to store the final network-state in
        """
        super(ValueAgent, self).__init__(sequences, network_object, refinement, network_path)
        self.net(self.num_seqs)

    def select(self, state):
        """
        select the next action according to the actual network state
        needed to override the parent method
        :param state: state to select next sequence from in non-linearized form (just as permutation)
        :return: selected action as index of next sequence
        """
        return self.act(linearize_state(state, self.num_seqs))

    def act(self, state, available=None):
        """
        Chooses action for provided state.
        :param state: state to perform action in
        :param available: vector containing the actions that lead to non-zero reward in this step in one-hot encoding
        :returns: the action (or accordingly the sequence) to select (align) next
        """
        net_output = self.net.forward(state)
        output_min = net_output.min()
        if output_min < 0:
            net_output -= output_min

        action = (net_output * (torch.tensor(available) if available is not None else 1)).max(0)[1].item()

        # in case of iterative refinement, shift the actions accordingly
        if self.refinement:
            action -= 1

        return action
