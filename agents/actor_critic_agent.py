import torch
from torch.distributions import Categorical

from agents.network_solver import NetworkSolver
from networks.actorcritic_networks import GenericActorCriticNetwork
from utils.utils import linearize_state


class ActorCriticAgent(NetworkSolver):
    def __init__(self, sequences, network_object: GenericActorCriticNetwork, refinement=False,
                 network_path='networks/ACNN.txt'):
        """
        Initialize the agent to train and select the alignment steps using an actor-critic algorithm
        :param sequences: sequences to align
        :param network_object: network-object to use for training and function approximation in the algorithm
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param network_path: filepath to store the final network-state in
        """
        super().__init__(sequences, network_object, refinement, network_path)

    def select(self, state):
        """
        select the next action according to the actual network state
        needed to override the parent method
        :param state: state to select next sequence from in non-linearized form (just as permutation)
        :return: selected action as index of next sequence
        """
        return self.act(linearize_state(state, self.num_seqs))[0]

    def act(self, state, available=None):
        """
        Chooses action for provided state.
        :param state: state to perform action in
        :param available: vector containing the actions that lead to non-zero reward in this step in one-hot encoding
        :returns: the action, the action value and the log-probability of the action
        """
        # forward the given state to receive state and action values
        value, policy_dist = self.net.forward(state)

        # select the action to perform in the given state
        probs = policy_dist.detach() * (torch.tensor(available) if available is not None else 1)

        if probs.min() == probs.max() and probs.min() < 1e-10:
            probs = torch.FloatTensor([0.25] * list(probs.size())[0]) * (torch.FloatTensor(available) if available is not None else 1.0)

        action = Categorical(probs).sample().item()
        # compute the log_prob auf the selected action
        log_prob = torch.log(policy_dist.squeeze(0)[action])

        if self.refinement:
            action -= 1

        return action, value.item(), log_prob

    def get_state_estimate(self, state):
        """
        get estimate of actual state based on previous learning
        :param state: actual state to estimate
        :return: state estimate
        """
        return self.net.forward(linearize_state(state, self.num_seqs))[0]
