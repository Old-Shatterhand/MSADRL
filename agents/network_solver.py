from agents.solver import Solver
from networks.network import GenericNetwork
from utils.utils import linearize_state


class NetworkSolver(Solver):
    def __init__(self, sequences, network_object: GenericNetwork, refinement=False,
                 network_path="networks/Example.txt"):
        """
        initializing an agent for the problem state using a network
        :param sequences: sequences to find the perfect multiple alignment of
        :param network_object: kind of network to use for the training
        :param network_path: path to store the network at
        """
        super().__init__(sequences=sequences, refinement=refinement)
        self.net = network_object(self.num_seqs)
        self.load = network_path is None
        self.network_path = network_path

    def get_network_path(self):
        """
        return the network-path to load/store the final state of the network in
        :return: network-path as specified in the __init__
        """
        return self.network_path

    def get_network(self):
        """
        access the network used in the actual instance of a solver
        :return: network used by this network-solver instance
        """
        return self.net

    def get_state_estimate(self, state):
        """
        get estimate of actual state based on previous learning
        :param state: actual state to estimate
        :return: state estimate
        """
        return self.net.forward(linearize_state(state, self.num_seqs)).max(0)[0].item()
