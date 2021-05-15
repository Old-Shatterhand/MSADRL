class Solver:
    """
    Every class that want to solve the problem should extend this class or
    at least should implement the two methods "name" and "select"
    """

    def __init__(self, sequences, refinement=False):
        """
        Initialize the profile and store all sequences that can be used in the alignment-process
        :param sequences: sequences to align optimal using the solver
        """
        if refinement:
            self.num_seqs = sequences.size() + 1
            self.profile = sequences
            self.sequences = self.profile.get_sequences()
        else:
            self.num_seqs = len(sequences)
            self.sequences = sequences

        self.refinement = refinement
        self.count = -1
        self.load = False

    def name(self):
        """
        Return the name of the agent actually used
        :return: name of actual agent
        """
        return self.__class__.__name__

    def select(self, state):
        """
        query the next step of aligning, therefore the actual profile is provided
        :param state: actual permutation of sequences
        :return:
        """
        self.count += 1
        return self.count

    def get_state_estimate(self, state):
        """
        get estimate of actual state based on previous learning
        :param state: actual state to estimate
        """
        pass

    def get_sequences(self):
        """
        access the sequences of the actual problem instance
        :return: sequences to align
        """
        return self.sequences

    def get_input_size(self):
        """
        method to query the number of sequences to align, used, with some additional computations
        as input-size for a network or used in the computation hash-values in table-agents
        :return: number of sequences to align
        """
        return len(self.sequences)

    def get_network(self):
        """
        return the actual network instance used in this agent if there is a network used
        :return: used network of the agent
        """
        return None

    def get_network_path(self):
        """
        return the path where to store the actual trained network
        :return: file-path of to store the network in
        """
        return None
