import numpy as np

from agents.solver import Solver
from utils.hash_q_table import HashQTable
from utils.utils import linearize_state


class TableSolver(Solver):
    def __init__(self, sequences, refinement=False):
        """
        initializing a table agent for the problem
        :param sequences: sequences to find the perfect multiple alignment of
        """
        super().__init__(sequences, refinement)
        self.num_seqs = len(self.sequences)
        self.table = HashQTable(self.num_seqs)

    def name(self):
        """
        Define the name according to the other names
        :return: name of this agent
        """
        return "TableAgent"

    def select(self, state):
        """
        Override the select-method from the super-class (needed for final alignment)
        :param state: state of alignment in non-linearized permutation form
        :return: action to select in the actual state
        """
        return self.act(linearize_state(state, self.num_seqs))

    def act(self, state, available=None):
        """
        Compute the action according to the actual table state
        :param state: actual state of multiple alignment in linearized form
        :param available: vector containing the actions that lead to non-zero reward in this step in one-hot encoding
        :return: index of sequence to select next
        """
        q_vals = self.table[state]
        min_val = min(q_vals)
        if min_val < 0:
            q_vals -= min_val
        action = (q_vals * (np.array(available) if available is not None else 1)).argmax(0)

        if self.refinement:
            action -= 1

        return action

    def get_state_estimate(self, state):
        """
        get estimate of actual state based on previous learning
        :param state: actual state to estimate
        :return: state estimate
        """
        return self.table[state].argmax()
