from utils.alignment import *
from utils.constants import SP_SCORE
from utils.hash_align_table import HashAlignTable


class AlignmentWrapper:
    def __init__(self, sequences, solver, score=SP_SCORE):
        """
        Control the alignment of a set of sequences performed by a Network
        :param sequences: sequences to be aligned
        :param solver: agent to use for selecting the order of the alignments
        """
        # initialize all fields and some additional utility classes for different supporting techniques during learning
        self.sequences = sequences
        self.num_seqs = len(self.sequences)
        self.solver = solver
        self.score = score
        self.permutation = []

        # hash table for faster alignments
        self.align_table = HashAlignTable(sequences)

        # vector containing binary encoding of valid actions
        self.available = np.ones(self.num_seqs)

        # store the best alignment found in the whole lifetime of the environment
        self.best_alignment = Profile([]), None

    def get_possible_moves(self):
        """
        return the available ove from the actual state of the alignment
        :return: available states in one-hot-encoding
        """
        return self.available

    def step(self, action):
        """
        Performs the selected action in the environment
        That means, the permutation is enlarged by the sequence selected in the last action
        :param action: action as the number of the sequence to align next to the actual state
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """

        # action already performed
        if action in self.permutation:
            return (-1000, 0), self.permutation, None, True

        self.permutation.append(action)
        self.available[action] = 0

        # stop the selection if there is only one sequence left to select...
        if len(self.permutation) == self.num_seqs - 1:
            # ... and add the missing sequence to the permutation
            self.permutation.append(list(set(range(self.num_seqs)) - set(self.permutation))[0])

        profile = align_progressive(self.permutation, self.sequences, align_table=self.align_table)

        # eventually update the best found alignment
        if len(self.permutation) == self.num_seqs:
            self.best_alignment, _ = compare_alignments(self.best_alignment, (profile, self.permutation), self.score)

        return profile.score(), self.permutation, profile, len(self.permutation) == self.num_seqs

    def run(self):
        """
        compute the alignment that the solver would perform
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        reward, permutation, profile, done = self.step(self.solver.select(self.permutation))
        while not done:
            reward, permutation, profile, done = self.step(self.solver.select(self.permutation))
        return reward, permutation, profile, done

    def reset(self):
        """
        reset the wrapper and all necessary classes to perform a new alignment which is than returned
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        self.soft_reset()
        reward, permutation, profile, done = self.run()
        if profile is not None:
            return reward, permutation, profile, done
        else:
            # perform in case of failed network/table-alignment an alignment in the order of the input sequences
            return (0, 0), None, Profile([]), False

    def soft_reset(self):
        """
        reset the wrapper and all necessary classes to perform a new alignment but without actually performing this
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        self.permutation = []
        self.available = np.ones(self.num_seqs)
        return (0, 0), [], None, False

    def evaluate_training(self):
        """
        Evaluate the training of the agent based on its ability to perform an alignment from scratch after the training
        and the best ever found alignment during learning
        :return: best alignment, final alignment
        """
        return self.best_alignment, self.reset()


class RefinementWrapper:
    def __init__(self, profile, solver, score):
        """
        Control the process of refining an already computed alignment
        :param profile: profile to start the refinement from
        :param solver: algorithm to use for refinement
        """
        self.num_seqs = profile.size()
        self.init_profile = profile
        self.profile = copy(profile)
        # print("START:\n" + str(self.profile))
        self.sequences = profile.get_sequences()
        self.score = score
        self.solver = solver

        self.action_list = []
        self.available = list(range(self.num_seqs + 1))

        self.align_table = HashAlignTable(self.init_profile)

        # store the best alignment found in the whole lifetime of the environment
        self.best_alignment = Profile([]), None

    def step(self, action):
        """
        Perform a step/action in the refinement process
        :param action: action to realign next
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        # print("ACTION:", action)
        # -1 as action indicates that the agent stops aligning
        if action == -1:
            self.best_alignment, _ = compare_alignments(self.best_alignment, (self.profile, self.action_list),
                                                        self.score)
            return self.profile.score(), self.action_list, self.profile, True

        # perform the action and realign the according sequence
        self.action_list.append(action)
        self.profile = align_iterative(self.action_list, self.profile, self.align_table)

        # compare for the best alignment and return the actual state
        self.best_alignment, _ = compare_alignments(self.best_alignment, (self.profile, self.action_list), self.score)
        return self.profile.score(), self.action_list, self.profile, len(self.action_list) == self.num_seqs

    def run(self):
        """
        perform the refinement and return the result
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        reward, action_list, profile, done = self.step(self.solver.select(self.action_list))
        while not done:
            reward, action_list, profile, done = self.step(self.solver.select(self.action_list))
        return reward, action_list, profile, done

    def reset(self):
        """
        reset the wrapper and all necessary classes to perform a new alignment which is than returned
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        self.soft_reset()
        return self.run()

    def soft_reset(self):
        """
        reset the wrapper and all necessary classes without performing a new run
        :return:  - the score of the computed alignment or 0 if the alignment is not ready
                  - the permutation of the sequences, that lead to the actual alignment
                  - the profile computed (only returned if the aligning is finished and valid)
                  - bool flag whether the alignment is finished or not
        """
        self.action_list = []
        self.profile = copy(self.init_profile)
        return (0, 0), [], self.profile, False

    def evaluate_training(self):
        """
        evaluate the training by returning the best found alignment and performing a last final refinement
        :return:
        """
        return self.best_alignment, self.reset()
