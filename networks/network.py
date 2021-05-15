from torch import nn


class GenericNetwork(nn.Module):
    def __init__(self, num_seqs):
        """
        generic network to use for alignments
        :param num_seqs: number of sequences to align
        """
        super().__init__()
        self.num_seqs = num_seqs
        self.input_size = num_seqs * (num_seqs - 1)

    def forward(self, state):
        """
        forward state through the network (is overridden in each network-type (GenericXXXNetwork))
        :param state: state to forward
        :return: state and action
        """
        pass
