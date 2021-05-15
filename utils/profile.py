import os
from copy import copy

import numpy as np

from utils.constants import GAP
from utils.utils import score, score_sp_column, fasta_output, get_score, merge_permutation


class Profile:
    def __init__(self, seqs, permutation=None):
        """
        create a profile as representation of an alignment of sequences
        an empty profile is defined a containing 0 sequences
        :param seqs: sequences as a list to represent by this profile instance
        """
        self.seqs = seqs.seqs if isinstance(seqs, Profile) else seqs
        self.default = self.seqs == []

        self.total_score = 0
        self.exact_matches = 0

        self.permutation = list(range(len(self.seqs))) if permutation is None else permutation
        self.missing_seq = -1

        # if a score for the profile can be computed, compute this score according to the profile
        if self.size() >= 2:
            self.sp_scores = np.zeros(len(self.seqs[0]))
            self._compute_sp_scores()

    def __eq__(self, other):
        """
        Checks equality of two profiles based on their sequences
        :param other: profile to be compared with
        :return: similarity or not
        """
        return isinstance(other, Profile) and self.seqs == other.seqs

    def __len__(self):
        """
        length of the profile defined as the length of the sequences or 0 if profile is empty
        :return: length of this profile as specified
        """
        return 0 if len(self.seqs) == 0 else len(self.seqs[0])

    def __str__(self):
        """
        turn the profile into a string
        :return: string-representation of the profile
        """
        return "\n".join(self.seqs)

    def __getitem__(self, item):
        """
        get a column from the profile as a list
        :param item: index of column to return
        :exception index out of bounds if item is bigger then length
        :return: column from the profile
        """
        return [seq[item] for seq in self.seqs]

    def __copy__(self):
        """
        return a memory copy of this profile
        :return: memory copy of the profile
        """
        return Profile(copy(self.seqs))

    def get_sequences(self):
        """
        get the sequences represented by the profile
        :return:
        """
        return self.seqs

    def add_gap(self, index):
        """
        Adding a gap in the complete profile
        :param index: index of the gap insertion
        """
        # Add the gap
        for i in range(self.size()):
            self.seqs[i] = self.seqs[i][:index] + "-" + self.seqs[i][index:]

        # if necessary recompute the sp_score
        if self.size() >= 2:
            c_score, exact = score_sp_column([self.seqs[i][index] for i in range(self.size())])
            self.sp_scores = np.insert(self.sp_scores, index, c_score)
            self.total_score += c_score
            if exact:
                self.exact_matches += 1

    def size(self):
        """
        return the size of the profile
        the profiles size is here defined as the number of sequences
        :return: size (number of sequences) of the represented alignment
        """
        return len(self.seqs)

    def score(self):
        """
        get all available/specified scores of the profile
        :return: scores of the profile
        """
        # return self.exact_matches/len(self), self.total_score, self.exact_matches, len(self)
        if self.default:
            return float('-inf'), -0.1, -1, -1
        if len(self) == 0:
            return 0, 0, 0, 0
        return self.total_score, self.exact_matches / len(self), self.exact_matches, len(self)

    def alt_align(self, base, i):
        """
        score the alignment of a base against a specific position of the alignment
        :param base: base to align
        :param i: index to align at
        :return: score of aligning the base to this position
        """
        return sum([score(base, seq[i]) for seq in self.seqs])

    def align(self, base, i):
        """
        score the alignment of a base against a specific position of the alignment
        :param base: base to align
        :param i: index to align at
        :return: score of aligning the base to this position
        """
        return sum([score(base, seq[i]) for seq in self.seqs]) + self.sp_scores[i]

    def get_and_remove_seq(self, index):
        """
        get and remove a sequence from the alignment
        :param index: index of sequence in original profile to be realigned and therefor extracted and removes
        :return: sequence and resulting permutation of this profile
        """
        index = self.permutation.index(index)
        seq = self.seqs.pop(index)
        self.permutation += [self.permutation.pop(index)]
        self._compute_sp_scores()
        return seq, self.permutation

    def set_permutation(self, permutation):
        """
        set the represented permutation of this profile
        :param permutation: permutation to be represented by the alignment (used for refinement of alignments)
        """
        self.permutation = permutation

    def store(self, folder, i, config, benchmark_name, sequence_names, permutation):
        """
        Store this alignment into a fasta formatted file
        :param folder: folder to store in
        :param i: number of file to store
        :param config: configuration of the agent to be stored
        :param benchmark_name: name of the benchmark
        :param sequence_names: names of the sequences
        :param permutation: permutation
        :return:
        """
        if not os.path.isdir(folder):
            print("No such directory:", folder + ".", "Please check the folder")
            return
        name = str(i) + "_" + config.name + "_" + get_score(config) + "_" + \
            ("R" if config.refinement else "P") + "_" + benchmark_name + ".tfa"
        print("Saving", name)
        with open(os.path.join(folder, name), 'w') as output:
            for i, seq in enumerate(self.seqs):
                output.write(sequence_names[permutation[i]] + "\n")
                output.write(fasta_output(seq) + "\n\n")
            output.flush()
            output.close()

    def _compute_sp_scores(self):
        """
        private method to compute the sum-of-pairs score for this profile
        """

        if len(self) > 0:
            length = len(self.seqs[0])
            for i in range(1, len(self.seqs)):
                if len(self.seqs[i]) != length:
                    print("Invalid profile: Sequences are not of same length")
                    return

        for i in range(len(self)):
            c = self[i]

            # compute the score of each column
            c_score, exact = score_sp_column(c)

            # update the scores
            self.sp_scores[i] = c_score
            self.total_score += c_score

            # count the exact matching columns
            if exact:
                self.exact_matches += 1


class StoreProfile:
    def __init__(self, profile, permutation):
        """
        Class to hold profiles' minimal information to be recreated from sequence, i.e. the permutation of the original
        sequences and the position of the gaps in this profile. This is far more efficient than storing the whole
        profile and is majorly used in HashAlignTables were most of the profiles are stored
        :param profile: profile to be represented
        :param permutation: permutation represented by this profile
        """
        self.gaps = [[pos for pos, char in enumerate(seq) if char == GAP] for seq in profile.get_sequences()]
        self.permutation = permutation

    def to_profile(self, seqs):
        """
        convert this minimal profile into a real profile based on the given sequences
        :param seqs: sequences to convert into a profile
        :return: profile based on the gaps in this profile
        """
        # special treatment if it is to refine alignments
        if isinstance(seqs, Profile):
            permutation = merge_permutation(seqs.size(), self.permutation)
            tmp_seqs = list(zip(*sorted(list(filter(lambda x: x[0] in permutation, enumerate(seqs.get_sequences()))),
                                        key=lambda x: permutation.index(x[0]))))[1]
            tmp = [insert_gaps(seq.replace(GAP, ""), gap_pos) for seq, gap_pos in zip(tmp_seqs, self.gaps)]
            return Profile(tmp)

        # normal treatment if it is normal alignment
        seqs = list(zip(*sorted(list(filter(lambda x: x[0] in self.permutation, enumerate(seqs))),
                                key=lambda x: self.permutation.index(x[0]))))[1]
        profile = Profile([insert_gaps(seq, gap_pos) for seq, gap_pos in zip(seqs, self.gaps)])
        return profile


def insert_gaps(seq: str, gaps):
    """
    Insert the gaps into the sequences at the positions
    :param seq: sequences to extends by the gaps
    :param gaps: positions of the gaps to insert
    :return: sequences with inserted gaps
    """
    for pos in gaps:
        seq = seq[:pos] + GAP + seq[pos:]
    return seq
