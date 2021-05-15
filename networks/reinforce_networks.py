from networks.ffnn_networks import PolicyFFNN, ValueFFNN, TinyValueFFNN, TinyPolicyFFNN
from networks.network import GenericNetwork


class GenericREINFORCENetwork(GenericNetwork):
    def __init__(self, num_seqs):
        """
        Initialize the network
        :param num_seqs: number of sequences to align in the actual instance of the problem
        """
        super().__init__(num_seqs)

    def forward(self, state):
        """
        forward state through the network
        :param state: state to forward
        :return: state and action
        """
        return self.value_net.forward(state), self.policy_net.forward(state)


class NoBaselineREINFORCENetwork(GenericREINFORCENetwork):
    """
    Network used for REINFORCE without a baseline
    """
    def __init__(self, num_seqs):
        super(NoBaselineREINFORCENetwork, self).__init__(num_seqs)

        self.policy_net = PolicyFFNN(num_seqs)

    def forward(self, state):
        """
        forward state through network
        :param state: state to forward
        :return: action values for the state the 1 is just a place holder to have common return types
        """
        return 1, self.policy_net.forward(state)


class REINFORCENetwork(GenericREINFORCENetwork):
    def __init__(self, num_seqs):
        super(REINFORCENetwork, self).__init__(num_seqs)

        # value net for baseline of REINFORCE
        self.value_net = ValueFFNN(num_seqs, 1)

        # policy net for action selection
        self.policy_net = PolicyFFNN(num_seqs)


class TinyREINFORCENetwork(GenericREINFORCENetwork):
    def __init__(self, num_seqs):
        super(TinyREINFORCENetwork, self).__init__(num_seqs)

        # value net for baseline of REINFORCE
        self.value_net = TinyValueFFNN(num_seqs, 1)

        # policy net for action selection
        self.policy_net = TinyPolicyFFNN(num_seqs)
