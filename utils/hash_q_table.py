import numpy as np

from utils.utils import hash_state


class HashQTable:
    def __init__(self, num_seqs):
        """
        Initialize the Q-Table as a wrapper around a dictionary. The reason is that the state-space is (n^(n+1)-1)/(n-1)
        But these many states are never visited in the small amount of games performed in training, so it is reduced to
        a dictionary with a perfect hash-function. So this class simulates a Q-table but only contains the already
        visited states and shrinks therefore the space and computational time needed for updating and querying
        The hash-function is only applicable to 36 sequences at maximum
        because the encoding as integers is not specified for larger instances
        :param num_seqs: number of sequences in the problem state the HashQTable is used in
        """
        if num_seqs > 36:
            raise ValueError("Cannot handle more than 36 sequences")
        self.num_seqs = num_seqs
        self.table = dict()

    def __getitem__(self, item):
        """
        Eventually create the item queried according to the structure this class is simulating
        :param item: state that is queried
        :return: array of action-values for the given state
        """
        # compute the hash-function of the state
        h_state = hash_state(item, self.num_seqs)

        # insert the state into the table if not already exists
        if h_state not in self.table:
            self.table[h_state] = np.zeros(self.num_seqs)
        return self.table[h_state]
