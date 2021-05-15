import itertools
import math

from utils.profile import Profile, StoreProfile
from utils.utils import hash_state_fast, compare_alignments


class HashAlignTable:
    def __init__(self, seqs):
        """
        HashTable to store and lookup alignments computed in the alignment process to not recompute these alignments
        :param seqs: number of sequences in the instance of the problem
        """
        self.seqs = seqs
        self.num_seqs = seqs.size() if isinstance(seqs, Profile) else len(seqs)
        self.table = dict()
        self.max = math.factorial(self.num_seqs)
        self.count = 0
        self.stats = [0 for _ in range(self.num_seqs)]

    def get(self, permutation, cutoff=2):
        """
        get the profile according to the permutation in the actual instance of the problem
        if original permutation not contained try sub-permutation iteratively
        by removing the last sequence from the permutation
        :param permutation: query-permutation
        :param cutoff: minimal number of sequences in permutations list, that should be searched for
        :return: empty profile if nothing found, otherwise return largest applicable profile
        """
        # linearize and compute the hash-value of the actual permutation
        state_hash = hash_state_fast(permutation, self.num_seqs)

        # search for permutation
        while state_hash not in self.table and len(permutation) > cutoff:
            # if not found, shorten the permutation
            permutation = permutation[:-1]
            state_hash = hash_state_fast(permutation, self.num_seqs)

        # return the according profile or the empty profile
        return (self.table[state_hash].to_profile(self.seqs), len(permutation)) \
            if state_hash in self.table else (Profile([]), 0)

    def set(self, permutation, profile):
        """
        set a profile for a new value
        :param permutation: permutation of sequences in the actual instance of the problem
        :param profile: profile to assign to this permutation
        """
        # if the permutation has maximal length, increase the count therefore
        self.count += len(permutation) == self.num_seqs
        self.stats[len(permutation) - 1] += 1

        # linearize and compute the hash-value of the actual permutation
        state_hash = hash_state_fast(permutation, self.num_seqs)

        if state_hash not in self.table:
            self.table[state_hash] = StoreProfile(profile, permutation)

    def is_full(self):
        """
        Return if the table holds all possible alignments, represented as the check if all possible leafs are contained
        :return: true if all possible permutation are contained
        """
        return self.count == self.max

    def get_best(self, score):
        """
        Get the best profile from the leafs of this tree/table
        :param score: score to use to find the best profile
        :return: best profile according to the specified score and its permutation
        """
        best, best_permutation = Profile([]), []

        for permutation in itertools.permutations(list(range(len(self.seqs)))):
            permutation = list(permutation)
            if len(best_permutation) == 0:
                best, changed = self.get(permutation)[0], True
                best_permutation = permutation
            else:
                (best, best_permutation), _ = compare_alignments((best, best_permutation),
                                                                 (self.get(permutation)[0], permutation), score)
        return best, best_permutation
