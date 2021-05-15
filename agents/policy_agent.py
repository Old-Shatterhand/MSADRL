import torch
from torch.distributions import Categorical

from agents.network_solver import NetworkSolver
from networks.reinforce_networks import GenericREINFORCENetwork
from utils.utils import linearize_state


class PolicyAgent(NetworkSolver):
    def __init__(self, sequences, network_object: GenericREINFORCENetwork, refinement=False,
                 network_path="networks/FFNN.txt"):
        """
        Initialize the agent to train and select the alignment steps following the REINFORCE algorithm
        :param sequences: sequences to align
        :param network_object: network to use for function approximation
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param network_path: filepath to store the network in
        """
        super().__init__(sequences, network_object, refinement, network_path)

    def select(self, state):
        """
        select the next action according to the actual network state
        needed to override the parent method
        :param state: state to select next sequence from in non-linearized form (just as permutation)
        :return: selected action as index of next sequence
        """
        action = self.net.forward(linearize_state(state, self.num_seqs))[1].argmax().item()

        if self.refinement:
            action -= 1

        return action

    def act(self, state, available=None):
        """
        Chooses action for provided state.
        :param state: state to perform action in
        :param available: vector containing the actions that lead to non-zero reward in this step in one-hot encoding
        :returns: the action, the action value and the log-probability of the action
        """
        # compute the probabilities
        value, probs = self.net.forward(state)

        # select an action out of all possibles
        tmp = probs * (torch.FloatTensor(available) if available is not None else 1.0)
        if tmp.min() == tmp.max() and tmp.min() < 1e-10:
            tmp = torch.FloatTensor([0.25] * list(tmp.size())[0]) * (torch.FloatTensor(available) if available is not None else 1.0)
        action = Categorical(torch.Tensor(tmp)).sample().item()
        
        # compute the logarithmic probability of this action and return both
        log_prob = torch.log(probs[action])

        if self.refinement:
            action -= 1

        return action, value, log_prob
