import numpy as np

from agents.network_solver import NetworkSolver
from networks.alphazero_networks import TinyA0_Network
from utils.utils import linearize_state


class AlphaZeroAgent(NetworkSolver):
    def __init__(self, sequences, network_object: TinyA0_Network, refinement=False, network_path='networks/A0.txt'):
        """
        AlphaZero-Agent holding methods for the training of an AlphaZero-Agent
        :param sequences: sequence to align in this agent
        :param network_object: network-object to use to learn the function approximation in the algorithm
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param network_path: filepath to store the final network-state in
        """
        super().__init__(sequences, network_object, refinement, network_path)

    def select(self, state):
        """
        Select an action based on the input state
        :param state: non-linearized state to select the action for
        :return: action selected based on the probabilities got from the net
        """
        probs = self.net.forward(linearize_state(state, self.num_seqs))[1].detach().numpy()
        action = np.random.choice(range(self.num_seqs), p=probs)

        if self.refinement:
            action -= 1

        return action

    def get_state_estimate(self, state):
        """
        Estimate the state based on the actual network-state using its state-estimation output-neuron
        :param state: state to estimate
        :return: estimation of the expected reward form that state on
        """
        return self.net.forward(linearize_state(state, self.num_seqs))[0]
