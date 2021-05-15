from os import listdir

from utils.alignment import align_seq_seq as align
from utils.constants import SP_SCORE
from utils.utils import read_fasta_data

ox_path = "../Databases/MDSA_all/oxbench/oxbench_mdsa_all"


def select_benchmark_files(folder, score, count_filter, sim_filter=None):
    """
    Read all files in the given folder according to the fasta-file-format and select data about the sequence count and
    the number of sequences in that file to evaluate the usability as a benchmark or optimization sequence file
    :param folder: folder holding all the sequence files
    :param score: score to optimize for
    :param count_filter: filter method to select the files according to their properties to shrink the number of outputs
    :param sim_filter: filtering method to select from possible benchmarks based on their pairwise similarities
    :return: candidates to be selected as benchmarks for the project
    """
    benchmarks = []
    for file in listdir(folder):
        sequences = read_fasta_data(folder + "/" + file)
        length, min_l, max_l, avg_l = 0, 10000, 0, 0
        for seq in sequences:
            # compute the directly measurable properties
            length = len(seq)
            min_l = min(min_l, length)
            max_l = max(max_l, length)
            avg_l += length
        avg_l //= len(sequences)

        # if filter based on static constraints is matched, check for similarity of the sequences
        if count_filter(len(sequences), min_l, max_l, avg_l):
            # if no similarity check is defined, add directly...
            if sim_filter is None:
                benchmarks.append(((file, min_l, max_l, avg_l, len(sequences)), ("-", "-", "-")))
            # ...otherwise compute the similarities and check the filter
            else:
                similarity = compute_pairwise_similarities(sequences, sim_filter, score)
                if similarity is not None:
                    benchmarks.append(((file, min_l, max_l, avg_l, len(sequences)), similarity))
    return benchmarks


def compute_pairwise_similarities(sequences, sim_filter, score):
    """
    Compute all pairwise similarities of the given sequences and check every pair for the filter to be fulfilled
    :param sequences: sequences to be checked
    :param sim_filter: filter to be applied on the pairwise similarities
    :param score: score to optimize for
    :return: (only returned if all pair pass the filtering) min, max and avg score of the computed pairwise scores
    """
    scores = []
    for a in range(len(sequences)):
        for b in range(a + 1, len(sequences)):
            cs = align(sequences[a], sequences[b]).score()[score]
            if sim_filter(cs):
                scores.append(cs)
            else:
                return None
    return min(scores), max(scores), sum(scores) / len(scores)


ben = select_benchmark_files(ox_path, SP_SCORE, lambda count, min_l, max_l, avg_l: 3 <= count <= 5 and avg_l < 150)
print("MinL\tMax:\tAvg:\tCount:\tMin-Sim:\tMax-Sim:\tAvg-Sim:\tFile:")
for b, s in ben:
    print(F"{b[1]}\t\t{b[2]}\t\t{b[3]}\t\t{b[4]}\t\t{round(s[0], 2)}\t\t{round(s[1], 2)}\t\t{round(s[2], 2)}\t\t{b[0]}")
