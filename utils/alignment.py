import itertools
from copy import copy
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

from utils.configurations import from_dict
from utils.constants import LEFT, DIAG, UP, seq_files, names, score_protein_gap
from utils.hash_align_table import HashAlignTable
from utils.profile import Profile
from utils.utils import *

"""
Dear programmer:
When I wrote this code, only god ans I knew how it worked. Now only god knows it!

Therefore, if you are trying to optimize these routines and it fails (most surely),
please increase this counter as a warning for the next person:
total_hours_wasted_here = 41
"""


def similarity_matrix(files):
    """
    print the simiarlity matrix as prettytable
    :param files: files to analyse
    """
    for seq_file in files:
        # take file from internal notation if given like this
        if isinstance(seq_file, int):
            seq_file = seq_files[seq_file]

        # read sequence files
        seqs, header = read_fasta_data(seq_file)

        matrix = PrettyTable()
        matrix.title = seq_file
        matrix.field_names = ["Name"] + [str(i + 1) for i in range(len(seqs))]

        # add the computed rows into a prettytable and print it
        identities = compute_similarity_matrix(seqs, header, True, 1, 0, 0)
        for row in identities[0]:
            matrix.add_row(row)
        print("Average:", identities[1])
        print(matrix)


def compute_similarity_matrix(sequences, header, dna, score_match=1, score_mismatch=0, score_gap=0):
    """
    Compute the similarity matrix for each input file as the maximal number of perfect matched macromolecules
    :param sequences: sequences to compute the similarity table for
    :param header: header of the sequences
    :param dna: flag indicating to handle the sequences as dna sequences
    :param score_match: score for a sequence match in DNA sequences
    :param score_mismatch: score for a sequence mismatch in DNA sequences
    :param score_gap: score for a gap
    :return: matrix containing the values for each sequences as string ready for a prettytable print
    """
    matrix = []
    cum_ident = 0
    for i in range(len(sequences)):
        row = [header[i]]
        for j in range(len(sequences)):
            if i == j:
                row.append("1")
                continue
            profile, s = align_seq_seq(sequences[i], sequences[j], dna, None, score_match, score_mismatch, score_gap)
            identity = s / len(profile)
            cum_ident += identity
            row.append(str(round(identity, 2)))
        matrix.append(row)
    return matrix, round(cum_ident / (len(sequences) * (len(sequences) - 1)), 2)


def merge_profiles(profile1, profile2):
    """
    Merge two profiles that share the first sequence in common
    :param profile1: first profile to merge
    :param profile2: second profile to merge
    :return: merged profile
    """
    sequences = ["" for _ in range(profile1.size() + profile2.size() - 1)]
    col1, col2, j = 0, 0, 0
    while col1 < len(profile1) and col2 < len(profile2):
        p1_col = profile1[col1]
        p2_col = profile2[col2]

        # both leading sequences are equal
        if p1_col[0] == p2_col[0]:
            for i in range(len(p1_col)):
                sequences[i] += p1_col[i]
            for i in range(1, len(p2_col)):
                sequences[i + len(p1_col) - 1] += p2_col[i]
            col1 += 1
            col2 += 1

        # first leading sequence has additional gap
        elif p1_col[0] == GAP:
            for i in range(len(p1_col)):
                sequences[i] += p1_col[i]
            for i in range(1, len(p2_col)):
                sequences[i + len(p1_col) - 1] += GAP
            col1 += 1

        # second leading sequence has additional gap
        elif p2_col[0] == GAP:
            for i in range(len(p1_col)):
                sequences[i] += GAP
            for i in range(1, len(p2_col)):
                sequences[i + len(p1_col) - 1] += p2_col[i]
            col2 += 1
        i += 1

    return Profile(sequences)


def center_star(sequences):
    """
    perform the center-start algorithm on the sequences
    :param sequences: sequences to align
    :return: center-star alignment
    """
    # compute the most similar sequence and choose it as start-/center-sequence for the alignment
    matrix = compute_similarity_matrix(sequences, [""] * len(sequences), True,
                                       score_dna_match, score_dna_mismatch, score_dna_gap)[0]
    average_similarity = [sum([float(x) for x in matrix[i][1:]]) / len(sequences) for i in range(len(sequences))]
    start = average_similarity.index(max(average_similarity))

    # compute the alignment of each sequence to the start-/center-sequence ...
    profiles = [align_seq_seq(sequences[start], sequences[i], False, None,
                              score_dna_match, score_dna_mismatch, score_dna_gap)[0]
                for i in range(len(sequences)) if i != start]

    # ... and merge them
    while len(profiles) > 1:
        profiles = [merge_profiles(profiles[-2], profiles[-1])] + profiles[:-2]

    return profiles[0]


def realignment(configs):
    """
    realign benchmarks from best_benchs.json
    :param configs: to be realigned, list of triples of optimized score, alignment type and benchmark name
    """
    best = read_best_file(path="../best_benches.json")[0]
    for score, mode, key in configs:
        if mode == "Refinement":
            pass
        else:
            seqs, header = read_fasta_data(seq_files[names.index(key)])
            config = best[score][mode][key]
            align_progressive(config["Permutation"], seqs).store("./results/", 0, from_dict(config["Configuration"]),
                                                                 key, header, config["Permutation"])
        print("Realinged", key)


def align_seq_seq(seq_a, seq_b, dna, perm=None, score_match_loc=score_dna_match, score_mismatch_loc=score_dna_mismatch,
                  score_gap_loc=score_dna_gap):
    """
    align a sequence again an already aligned profile
    this code follows the Needleman-Wunsch-Algorithm (1970) and the review by Waterman (1976)
    :param seq_a: first sequence to align
    :param seq_b: second sequence to align
    :param dna: flag indicating that the sequences has to ba handles as dna sequences
    :param perm: permutation to use to solve conflicts in the direction-matrix
    :param score_match_loc: local scoring for a match between two bases or nucleotides
    :param score_mismatch_loc: local scoring for a mismatch between two bases or nucleotides
    :param score_gap_loc: local scoring for a gap between the two sequences
    :return: profile of the profile-sequence alignment
    """
    if perm is None:
        perm = [LEFT, DIAG, UP]
    s_matrix = np.zeros((len(seq_a) + 1, len(seq_b) + 1))
    d_matrix = np.zeros((len(seq_a) + 1, len(seq_b) + 1), dtype='int32')

    dna = seq_a[0].islower() or dna

    # fill the edges of the matrices
    for i in range(1, len(seq_a) + 1):
        s_matrix[i, 0] = i * score_gap_loc
        d_matrix[i, 0] = UP
    for j in range(len(seq_b) + 1):
        s_matrix[0, j] = j * score_gap_loc
        d_matrix[0, j] = LEFT

    # fill the matrix according to the algorithms
    for i in range(1, len(seq_a) + 1):
        for j in range(1, len(seq_b) + 1):
            # compute the possible scores for the alignment
            d = s_matrix[i - 1, j - 1] + ((score_match_loc if seq_a[i - 1] == seq_b[j - 1] else score_mismatch_loc)
                                          if dna else score(seq_a[i - 1], seq_b[j - 1]))
            l = s_matrix[i, j - 1] + score_gap_loc
            u = s_matrix[i - 1, j] + score_gap_loc

            # fill the matrix with the maximal value
            s_matrix[i, j] = max(d, l, u)

            # fill the according value from the computation into the direction-matrix
            if s_matrix[i, j] == d:
                d_matrix[i, j] += DIAG
            if s_matrix[i, j] == l:
                d_matrix[i, j] += LEFT
            if s_matrix[i, j] == u:
                d_matrix[i, j] += UP

    seq_a_idx, seq_b_idx = len(seq_a), len(seq_b)
    aligned_a, aligned_b = "", ""
    # Backtracking of the alignment through the Direction-Matrix and synchronously enlarging the sequences as needed
    while seq_a_idx != 0 or seq_b_idx != 0:
        for p in perm:
            # go diagonal in the alignment
            if p == DIAG and d_matrix[seq_a_idx, seq_b_idx] & DIAG != 0:
                seq_a_idx -= 1
                seq_b_idx -= 1
                aligned_a = seq_a[seq_a_idx] + aligned_a
                aligned_b = seq_b[seq_b_idx] + aligned_b
                break
            # go left in the alignment, i.e. insert a gap in the sequence
            elif p == UP and d_matrix[seq_a_idx, seq_b_idx] & UP != 0:
                seq_a_idx -= 1
                aligned_a = seq_a[seq_a_idx] + aligned_a
                aligned_b = '-' + aligned_b
                break
            # go up in the alignment, i.e. insert a gap in the existing profile
            elif p == LEFT and d_matrix[seq_a_idx, seq_b_idx] & LEFT != 0:
                seq_b_idx -= 1
                aligned_a = '-' + aligned_a
                aligned_b = seq_b[seq_b_idx] + aligned_b
                break
    return Profile([aligned_a, aligned_b]), s_matrix[len(seq_a), len(seq_b)]


def align_prof_seq(seq, prof, perm=None, score_gap_loc=score_dna_gap):
    """
    align a sequence again an already aligned profile
    this code extends the Needleman-Wunsch-Algorithm (1970) and
    the review by Waterman (1976) to a profile-sequence alignment
    :param seq: sequence to align
    :param prof: profile to align
    :param perm: permutation to use to solve conflicts in the direction-matrix
    :param score_gap_loc: local scoring for a gap between the two sequences
    :return: profile of the profile-sequence alignment
    """
    if perm is None:
        perm = [LEFT, DIAG, UP]
    s_matrix = np.zeros((len(seq) + 1, len(prof) + 1))
    d_matrix = np.zeros((len(seq) + 1, len(prof) + 1), dtype='int32')

    # fill the edges of the matrices
    for i in range(1, len(seq) + 1):
        s_matrix[i, 0] = i * score_gap_loc
        d_matrix[i, 0] = 4
    for j in range(1, len(prof) + 1):
        s_matrix[0, j] = s_matrix[0, j - 1] + prof.align(GAP, j - 1)
        d_matrix[0, j] = 2

    # fill the matrix according to the algorithms
    for i in range(1, len(seq) + 1):
        for j in range(1, len(prof) + 1):
            # compute the possible scores for the alignment
            d = s_matrix[i - 1, j - 1] + prof.align(seq[i - 1], j - 1)
            l = s_matrix[i, j - 1] + prof.align(GAP, j - 1)
            u = s_matrix[i - 1, j] + score_gap_loc * prof.size()

            # fill the matrix with the maximal value
            s_matrix[i, j] = max(d, l, u)

            # fill the according value from the computation into the direction-matrix
            if s_matrix[i, j] == d:
                d_matrix[i, j] += DIAG
            if s_matrix[i, j] == u:
                d_matrix[i, j] += UP
            if s_matrix[i, j] == l:
                d_matrix[i, j] += LEFT

    aligned_seq = ""
    a_seq_index = len(seq)
    a_prof_index = len(prof)
    # Backtracking of the alignment through the Direction-Matrix and synchronously enlarging the sequences as needed
    while a_seq_index >= 0 and a_prof_index >= 0 and a_seq_index + a_prof_index != 0:
        for i in perm:
            # go diagonal in the alignment
            if i == DIAG and d_matrix[a_seq_index, a_prof_index] & DIAG != 0:
                a_seq_index -= 1
                a_prof_index -= 1
                aligned_seq = seq[a_seq_index] + aligned_seq
                break
            # go left in the alignment, i.e. insert a gap in the sequence
            if i == LEFT and d_matrix[a_seq_index, a_prof_index] & LEFT != 0:
                a_prof_index -= 1
                aligned_seq = "-" + aligned_seq
                break
            # go up in the alignment, i.e. insert a gap in the existing profile
            if i == UP and d_matrix[a_seq_index, a_prof_index] & UP != 0:
                a_seq_index -= 1
                aligned_seq = seq[a_seq_index] + aligned_seq
                prof.add_gap(a_prof_index)
                break
    # return the new computed profile of the two profiles
    return Profile(prof.seqs + [aligned_seq])


def align_progressive(permutation, seqs, align_table=None, pw_perm=None, m_perm=None):
    """
    align the given sequences progressive using the specified permutation and the alignment-table
    :param permutation: permutation specifying the sequence how to align the sequences
    :param seqs: sequences to align
    :param align_table: hash-table to use in alignment to prevent computations of previous aligned sequences
    :param pw_perm: permutation to solve conflicts in the direction-matrix in the pairwise sequence alignment
    :param m_perm: permutation to solve conflicts in the direction-matrix in the sequence-profile alignment
    :return: profile of all aligned sequences
    """
    score_gap = score_dna_gap if seqs[0].islower() else score_protein_gap
    if len(permutation) < 2:
        return Profile([seqs[permutation[0]]])
    if pw_perm is None:
        pw_perm = [LEFT, DIAG, UP]
    if m_perm is None:
        m_perm = [LEFT, DIAG, UP]

    prof = Profile([])
    # find previous alignments...
    if align_table is not None:
        prof = align_table.get(permutation)[0]

    # otherwise initialize the alignment with the first two sequences
    if prof.size() == 0:
        prof, _ = align_seq_seq(seqs[permutation[0]], seqs[permutation[1]], False, pw_perm, score_gap_loc=score_gap)
        if align_table is not None:
            align_table.set(permutation[:2], prof)

    # align the remaining sequences in the given sequence
    start = max(2, prof.size())
    for i in range(start, len(permutation)):
        prof = align_prof_seq(seqs[permutation[i]], prof, m_perm, score_gap_loc=score_gap)
        if align_table is not None:
            align_table.set(permutation[:(i + 1)], prof)
    return prof


def align_iterative(actions, profile, align_table=None, perm=None):
    """
    Perform an iterative alignment of the sequences in the profile according to the selection-order defined by the
    actions list. The actions in the list refer to the indexes of the according sequences in the original list.
    :param actions: actions to perform while the iterative refinement
    :param profile: profile to perform the actions (profile after performing all actions in the list except the last one
    :param align_table: dynamic structure holding the sequences
    :param perm: permutation of the order in which to solve conflicts in the alignment processes
    :return: profile of the alignment of the sequences after applying the last action to the profile
    """
    score_gap = score_dna_gap if profile[0][0].islower() else score_protein_gap
    # set the default order to solve conflicts
    if perm is None:
        perm = [LEFT, DIAG, UP]

    # preprocess the action list by breaking it at the first -1, as this action stops the end of the refining
    action_list = copy(actions)
    if -1 in action_list:
        action_list = action_list[:action_list.index(-1)]

    # realigning a sequence two times in a row makes no sense,
    # so I delete those parts of the action-sequence from the list
    i = 1
    while i < len(action_list):
        if action_list[i] == action_list[i - 1]:
            del action_list[i]
        else:
            i += 1

    # if the remaining action list after preprocessing is empty, nothing to align left; so, return the input profile
    if len(action_list) == 0:
        return profile

    # query the permutation in the dynamic datastructure to save some computational time to not realign previous steps
    if align_table is not None:
        stored, count = align_table.get(action_list, 1)
        if count == len(action_list):
            return copy(stored)
        if count != 0:
            prof = copy(stored)
            prof.set_permutation(merge_permutation(profile.size(), action_list[:count]))
        else:
            prof = copy(profile)
    else:
        prof, count = copy(profile), 0

    # align the remaining part of the action list, that is new to the align-table
    for i in range(count, len(action_list)):
        seq, permutation = prof.get_and_remove_seq(action_list[i])
        prof = align_prof_seq(seq, prof, perm, score_gap_loc=score_gap)
        prof.set_permutation(permutation)
        if align_table is not None:
            align_table.set(action_list[:(i + 1)], copy(prof))
    return prof


def alignment(a, b, perm=None):
    """
    align two objects (profiles or sequences) against each other only one can be a profile
    profile-profile alignment is not supported since not needed for this approach
    :param a: first sequence/profile
    :param b: second sequence/profile
    :param perm: permutation for direction-matrix to use for alignment
    :return: profile of the alignment of the two sequences/profile
    """
    if perm is None:
        perm = [LEFT, DIAG, UP]
    if isinstance(a, Profile):
        return align_prof_seq(b, a, perm, score_gap_loc=(score_dna_gap if b[0].islower() else score_protein_gap))
    if isinstance(b, Profile):
        return align_prof_seq(a, b, perm, score_gap_loc=(score_dna_gap if a[0].islower() else score_protein_gap))
    return align_seq_seq(a, b, perm, score_gap_loc=(score_dna_gap if a[0].islower() else score_protein_gap))[0]


def brute_force_alignment(sequences, score, hash_table=None, file_name=None, notification=None):
    """
    Brute-Force-Alignment to track the results of the other alignments and find the best existing alignment
    !!!ONLY PROGRESSIVE ALIGNMENTS!!!
    :param sequences: sequences to align
    :param score: score to use for optimization
    :param file_name: file name the sequences are from
    :param notification: flag to notify the user
    :param hash_table: hash-table to use for aligning
    :return:
    """
    best_profile = None
    best_score = (0, 0, 0, 0)
    best_permutation = None
    if hash_table is None:
        hash_table = HashAlignTable(sequences)
    # iterate over all permutations possible
    permutations = list(itertools.permutations(list(range(len(sequences)))))
    print("Brute-Force Alignment")
    for i, permutation in enumerate(permutations):
        permutation = list(permutation)
        print("\r" + str(i), "/", len(permutations))
        profile = align_progressive(permutation, sequences, hash_table)

        # score the permutation
        a_score = profile.score()
        if a_score[score] > best_score[score]:
            best_score = a_score
            best_profile = profile
            best_permutation = permutation
    # return the best profile in aligned form, the according score
    # and the permutation of the sequence-indexes according to the input-list of sequences
    if file_name is not None and notification is not None:
        optimizing = "SP" if score == SP_SCORE else "C"
        message = F"Brute-Force result on {file_name}:\n" \
                  F"\tMode: Progressive, Optimizing: {optimizing}-Score\n" \
                  F"\tPermutation: {best_permutation}\n" \
                  F"\tbest Alignment found: {best_score[0]}, {round(best_score[1], 2)}\n"

        print(message)
        if notification:
            notify(message)

    return best_profile, best_score, best_permutation


def multiprocessing_brute_force(b_ids, args):
    # initialize the multithreading tools needed
    pool = ThreadPool(processes=min(len(b_ids), cpu_count() if args.Multi == 1 else args.Multi))
    tasks = [None for _ in range(len(b_ids))]
    names = [[] for _ in range(len(b_ids))]

    for i, seq_file in enumerate(b_ids):
        sequences, names[i] = read_fasta_data(seq_files[list(b_ids.keys())[i]])
        tasks[i] = pool.apply_async(brute_force_alignment, (sequences, args.Optimize, None,
                                                            os.path.basename(seq_files[list(b_ids.keys())[i]]).
                                                            split(".")[0], args.Notify))

    for i, seq_file in enumerate(b_ids):
        profile, score, permutation = tasks[i].get()

        if args.Folder is not None:
            profile.store(args.Folder, i, args, os.path.basename(seq_file), names[i], permutation)
