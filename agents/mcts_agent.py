import time

import numpy as np
import torch

from agents.solver import Solver
from utils.alignment import align_progressive, align_iterative, center_star
from utils.constants import SP_SCORE
from utils.hash_align_table import HashAlignTable
from utils.profile import Profile
from utils.utils import get_sequences
from utils.wrapper import AlignmentWrapper, RefinementWrapper


def select_child(children):
    """
    Tree policy selecting of the provided children based on the maximal c or n
    c is computed using the UCB1 rule presented in Kocsis and Szepesvari in 2006
    the computation of c balances the exploration and exploitation
    :param children: children of a node to select from
    :return: the best/most promising child
    """
    return max(children, key=lambda node: node.u * (0 if node.state[-1] in node.state[:-1] else 1))


class MCTSAgent(Solver):
    """
    MCTS Agent to solve the alignment problem
    This agent uses UCTs (UCB applied to MCTS) to select actions which is a major improvement over vanilla MCTS
    """

    def __init__(self, sequences, simulations=0, rollouts=1, c=1, score=SP_SCORE, refinement=False, console=False,
                 adjust=True):
        """
        initialize the agent
        :param sequences: sequences to align
        :param simulations: number of simulations to make before selecting an action
        :param rollouts: number of rollouts to perform in each simulation
        :param c: UCB-Parameter to balance exploration/exploitation
        :param score: score to optimize for
        :param refinement: flag indicating to train for refinement
        :param console: flag indicating commandline outputs
        """
        super().__init__(sequences, refinement)
        self.state = []
        self.children = []
        self.align_table = HashAlignTable(Profile(sequences) if self.refinement else sequences)
        self.simulations = self.num_seqs * 50 if simulations == 0 else simulations
        self.rollouts = rollouts
        self.c = c
        self.score = score
        self.steps = 0
        self.console = console
        self.adjust = adjust and score == SP_SCORE
        if self.adjust:
            self.min_score = self.estimate_min()
            self.max_score = center_star(self.sequences).score()[SP_SCORE]
            if self.max_score < 0:
                self.max_score /= 2
            else:
                self.max_score *= 2

    def __str__(self):
        """
        ToString method for the tree to generate an overview of the actual state of MCTS
        :return: string representation of the tree
        """
        output = ""
        for child in self.children:
            output += F"{child.state}: ({child.n}, {child.v}, {child.u})\n"
        return "current MCTS-UCT (n, v, u)\n" + output

    def estimate_min(self):
        """
        Estimate minimal score for problem by taking the minimal score of 10 random rollouts form the initial node
        :return: estimated minimal score
        """
        return min([align_progressive(
            list(np.random.permutation(list(range(self.num_seqs - (1 if self.refinement else 0))))), self.sequences,
            self.align_table).score()[self.score] for _ in range(10)]) - 1

    def adjustment(self, score):
        """
        Adjust input score according to estimated minimum and computed maximum
        :param score: score to be adjusted
        :return: adjusted score
        """
        return (score - self.min_score) / (self.max_score - self.min_score)

    def set_align_table(self, align_table):
        """
        Set the used align-table to speed up alignments
        :param align_table: align-table to use
        """
        self.align_table = align_table

    def get_align_table(self):
        """
        Return the used and extended align-table
        :return: used and extended align-table
        """
        return self.align_table

    def set_state(self, state):
        """
        Setter method for the state that is placed in the root-node in the next search
        :param state: state to be used as root for next search
        """
        self.state = state

    def select(self, state):
        """
        main method of Monte-Carlo simulations selecting the next actions based on the given state
        :param state: state to select action in
        :return: action selected based on the MC-rollouts
        """
        self.set_state(state)

        self.init_tree()
        self.simulate()

        # select action
        action = max(self.children, key=lambda c: c.n).state[-1]
        # if self.console:
        print(F"Action {action} selected")
        return action

    def init_tree(self, stored=True):
        """
        Initialize the tree from the state of this agent.
        :param stored: determine if precomputed subtrees of the actual tree are reused or discarded
        """
        # get a subtree of the previous computation to improve the performance of the algorithm
        if stored:
            for child in self.children:
                if self.state == child:
                    self.children = child.children
                    self.steps = child.n
                    return
        # ... or generate all children from the state, also these that have an immediate reward of 0
        self.children = [MCTSNode(self.state + [i], self.num_seqs, self.score, self) for i in
                         (range(-1, self.num_seqs - 1) if self.refinement else range(self.num_seqs))]

    def simulate(self):
        """
        Simulate rollouts in the tree
        :return: average score gotten from the rollouts
        """
        # perform the simulations
        cum_game_score = 0
        for s in range(1, 1 + self.simulations):
            # select the children according to the implemented tree-policy
            node = select_child(self.children)
            game_score = node.rollout(s + self.steps)
            cum_game_score += game_score

            # update children's values
            for child in self.children:
                if child != node:
                    child.update_u(s + self.steps)
                else:
                    node.visit(game_score, s + self.steps)
        return cum_game_score / self.simulations

    def get_probabilities(self, state):
        """
        Compute a probability distribution over all available actions according to the alignments
        resulting from the actions. This is needed for AlphaZero-Learning
        :param state: state to find the PD for
        :return: PD over all (not only the valid) actions
        """
        self.set_state(state)
        self.init_tree(stored=False)
        score = self.simulate()
        return torch.nn.functional.softmax(torch.Tensor([child.n for child in self.children]), dim=0).numpy(), score

    def set_align_table(self, align_table):
        """
        Set the hash-table of previously computed alignments
        :param align_table: table to assign to own table
        """
        self.align_table = align_table

    def get_align_table(self):
        """
        Return the hash-table of previously computed alignments
        :return: alignment table computed/extended while computation
        """
        return self.align_table


class MCTSNode:
    """
    UCT-node to represent a partial alignment
    """

    def __init__(self, state, num_seqs, score, mcts_agent):
        """
        initialize the node
        :param state: state represented by the node
        :param num_seqs: number of sequences to align in total
        :param score: score to optimize for
        :param mcts_agent: parent agent, used to access algorithmic fields like
            the number of rollouts, the sequences, the UCB-parameter c and the align_table
        """
        self.state = state
        self.num_seqs = num_seqs
        self.children = []
        self.score = score
        self.mcts_agent = mcts_agent

        # search parameters
        self.n = 0
        self.u = 0
        self.v = 0
        self.total_score = 0

        # initialize fields to track state of the node (leaf / (tb.) expanded)
        self.is_leaf = self.num_seqs == len(self.state)
        self.to_expand = False
        self.is_expanded = False

        # compute the score of the alignment represented by this node if this node is a leaf that cannot be expanded
        self.scoring = 0
        if self.is_leaf and len(set(self.state)) == self.num_seqs:
            if self.mcts_agent.refinement:
                self.scoring = align_iterative(self.state, self.mcts_agent.profile,
                                               self.mcts_agent.align_table, ).score()[self.score]
            else:
                self.scoring = align_progressive(self.state, self.mcts_agent.sequences,
                                                 self.mcts_agent.align_table).score()[self.score]
            if self.mcts_agent.adjust:
                self.scoring = self.mcts_agent.adjustment(self.scoring)

    def __eq__(self, other):
        """
        Define the equality between nodes as equality of their represented states
        A node can also be equal to a list if this list is equal to the state of the node
        this is needed for tree-saving
        :param other: compared state
        :return: bool flag indicating equality or not
        """
        if isinstance(other, MCTSNode):
            return self.state == other.state
        elif isinstance(other, list):
            return self.state == other
        else:
            return False

    def rollout(self, step):
        """
        perform a rollout starting/continuing in this node/state
        :param step: number of games played before
        :return: the score reached in the rollout
        """
        # i this is a leaf return the represented score
        if self.is_leaf:
            return self.scoring
        # if this node has to be expanded, expand and continue
        if self.to_expand:
            self.expand()
        # if this node is already expanded or has just been expanded select the
        if self.is_expanded:
            # select a node according to the tree-policy and pass the rollout to this node
            node = select_child(self.children)
            rollout_score = node.rollout(step)

            # update the children of this node according to the received result
            for c in self.children:
                if c != node:
                    c.update_u(step)
                else:
                    node.visit(rollout_score, step)
        # if this is the first explicit, by the tree-policy selected visit after creation, perform a rollout
        else:
            rollout_score = 0
            for _ in range(self.mcts_agent.rollouts):
                tmp_state = self.state.copy()
                self.to_expand = True
                possible_actions = list(
                    range(-1, self.num_seqs - 1) if self.mcts_agent.refinement else set(range(self.num_seqs)) - set(
                        tmp_state))

                # append all possible actions in an proper way so that the result is not 0
                while len(tmp_state) < self.num_seqs:
                    action = np.random.choice(possible_actions)
                    tmp_state = tmp_state + [action]
                    if not self.mcts_agent.refinement:
                        possible_actions.remove(action)

                # compute the alignments score
                if len(tmp_state) == len(set(tmp_state)):
                    if self.mcts_agent.refinement:
                        tmp_score = align_iterative(tmp_state, self.mcts_agent.profile,
                                                    self.mcts_agent.align_table).score()[self.score]
                    else:
                        tmp_score = align_progressive(tmp_state, self.mcts_agent.sequences,
                                                      self.mcts_agent.align_table).score()[self.score]
                    if self.mcts_agent.adjust:
                        rollout_score += self.mcts_agent.adjustment(tmp_score)
        return rollout_score

    def visit(self, game_reward, step):
        """
        update the values of the node according to the changed properties in the tree
        :param game_reward: reward of the last performed game
        :param step: number of games already played
        """
        self.n += self.mcts_agent.rollouts
        self.total_score += game_reward
        self.v = self.total_score / self.n
        self.update_u(step)

    def update_u(self, step):
        """
        Update the u-field that is used by the tree-policy to determine which node to explore next
        :param step: number of games already played
        """
        self.u = self.v + self.mcts_agent.c * (np.sqrt(np.log(step) / (1 + self.n)))

    def expand(self):
        """
        expand the node
        """
        # check whether expansion is possible or already done
        if self.is_leaf or self.is_expanded:
            return
        self.is_expanded = True

        # add all children to the node
        self.children = [MCTSNode(self.state + [i], self.num_seqs, self.score, self.mcts_agent) for i in
                         (range(-1, self.num_seqs - 1) if self.mcts_agent.refinement else range(self.num_seqs))]

    def print(self, depth):
        """
        Print a string representation of this node and all of its children
        :param depth: depth of the node
        """
        for child in self.children:
            print('\t' * depth + F"{child.state}: ({child.n}, {child.v}, {child.u})")
            child.print(depth + 1)


if __name__ == "__main__":  # this file was run from the command line
    print("########################################################")
    print("##Starting training of a Monte-Carlo Tree-Search agent##")
    print("########################################################")
    print()

    score = SP_SCORE

    seqs = get_sequences(count=3, length=6, different=True)
    agent = MCTSAgent(seqs, rollouts=2, adjust=True)
    env = AlignmentWrapper(seqs, agent, score)

    start = time.time()
    reward, permutation, profile, _ = env.run()
    end = time.time()

    print(str(profile))
    print("Score:", reward[score], F"({permutation})")
    print("Trainer ran for %.2f seconds" % (end - start))

    start = Profile(["ctattg", "ctaccg", "ctatgt"])
    agent = MCTSAgent(sequences=start, refinement=True)
    env = RefinementWrapper(start, agent, score)

    start = time.time()
    reward, permutation, profile, _ = env.run()
    end = time.time()

    print(str(profile))
    print("Score:", reward[score], F"({permutation})")
    print("Trainer ran for %.2f seconds" % (end - start))
