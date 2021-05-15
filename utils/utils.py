import json
import math
import os
import random
import subprocess
import sys
from contextlib import contextmanager
from json import JSONDecodeError

import numpy as np
from prettytable import PrettyTable

from utils.constants import GAP, score_dna_match, score_dna_mismatch, score_dna_gap, PEPTIDE_SCORE_MATRIX, other, \
    sequences, TABLE_AGENT, REFERENCES, SP_SCORE


def merge_permutation(num_seqs, additive):
    """
    Merge the permutations from iterative refinement and the base permutation that ead to the starting points of the
    iterative refinements
    :param num_seqs: number of sequences
    :param additive: permutation to merge
    :return: merged
    """
    base = list(range(num_seqs))
    for a in additive:
        base.remove(a)
        base.append(a)
    return base


def compare_alignments(p1, p2, score):
    """
    Compare two alignments with each other based on the score they lead to and the score they are optimized for
    :param p1: first alignment to compare
    :param p2: second alignment to compare
    :return: better alignment, an case of equal quality, return the first one and True if new best alignment is found
    """
    if p1[0] is None and p2[0] is None:
        return (None, None, None), False
    if p1[0] is None:
        return p2, True
    if p2[0] is None:
        return p1, False
    p1_score = p1[0].score()
    p2_score = p2[0].score()
    if (p1_score[score], p1_score[1 - score]) < (p2_score[score], p2_score[1 - score]):
        return p2, True
    return p1, False


def score(base_a, base_b):
    """
    score two bases of amino-acids according to Fasta naming convention
    :param base_a: first base to score
    :param base_b: second base to score
    :return: score of the two bases according to the peptide matrix or the defined nucleotide scoring function
    """
    ord_a = ord(base_a)
    ord_b = ord(base_b)

    # dont score two gaps
    if ord_a + ord_b == 2 * ord(GAP):
        return 0
    # score nucleotides (DNA/RNA)
    elif ord_a + ord_b > 140:  # larger than 'a' and '-'
        return score_nucleotides(base_a, base_b)
    # score peptides (Proteins)
    else:
        return PEPTIDE_SCORE_MATRIX[ord_a - 65][ord_b - 65]


def score_nucleotides(base_a, base_b):
    """
    score a pair of bases
    :param base_a: first base
    :param base_b: second base
    :return: score according to the specified scores
    """
    if base_a != GAP and base_b != GAP:
        if base_a == base_b:
            return score_dna_match
        else:
            return score_dna_mismatch
    return score_dna_gap


def score_sp_column(col):
    """
    score a list of bases that can be taken from the column of a profile
    :param col: list of bases (column)
    :return: according sum-of-pairs score
    """
    c_score = 0
    exact = True
    for a in range(len(col)):
        for b in range(a + 1, len(col)):
            d_score = score(col[a], col[b])
            if d_score < score_dna_match:
                exact = False
            c_score += d_score
    return c_score, exact


def linearize_state(state, num_seqs):
    """
    linearize a state to be able to put it into a network or use in a table agent
    :param state: actual state to select an appropriate action for
    :param num_seqs: number of sequences in the actual multiple alignment
    :return: linearized state one-hot-encoded
    """
    state = state[:(num_seqs - 1)]
    output = [0] * (num_seqs * (num_seqs - 1))
    for i, v in enumerate(state):
        output[i * num_seqs + v] = 1
    return output


def linearize_complete_state(state, num_seqs):
    """
    linearize a complete state, not the shortened version as above
    :param state: state to be linearized
    :param num_seqs: number of sequences the state an contain at most
    :return: linearized form of the state
    """
    output = [0] * (num_seqs * len(state))
    for i, v in enumerate(state):
        output[i * num_seqs + v] = 1
    return output


def hash_state(state, num_seqs):
    """
    Hash-function as the heart and crucial part of this class

    Example using self.num_seqs=3:
    input:  [0,1,0,1,0,0]
    chunks: [[0,1,0],[1,0,0]]
    chars:  ['1','0'] => '10' in number_system with base 3
    h_val:  3

    :param state: state to be hashed
    :param num_seqs: number of sequences that are aligned
    :return: hash-value of input-state
    """
    # break the state into chunks, representing each one sequence selection in one-hot-encoding
    chunks = [state[i:i + num_seqs] for i in range(0, len(state), num_seqs)]

    # computing the char-representation of each char
    chars = [chunk_to_char(chunk) for chunk in chunks]

    # compute the resulting hash-value
    hash_val = int(''.join(chars), base=num_seqs + 1)
    return hash_val


def chunk_to_char(chunk):
    """
    convert a part of a linearized state into a char representation
    used to hash a state for the HashQTable
    only applicable if number of sequence is below 37 that the behaviour is not specified
    :param chunk: part of the linearized alignment
    :return: encoding a char
    """
    maxi = np.argmax(chunk, 0) + 1
    return chr(48 + maxi) if maxi < 10 else chr(55 + maxi)


def hash_state_fast(permutation, num_seqs):
    """
    directly hash a permutation that is not linearized
    useful for hash align table
    :param permutation: permutation to be hashed
    :param num_seqs: number of sequences the permutation can at most contain
    :return: hash-value for a permutation
    """
    return int(''.join([chr(49 + p) if p < 9 else chr(56 + p) for p in permutation]), base=num_seqs + 1)


def array_softmax(array, theta=1):
    """
    Applies the softmax-function to the given array with determinism parameter theta
    :param array: array to apply softmax to
    :param theta: determinism-parameter, will be multiplied with the array
    :return: probability distribution from the array and parameter theta
    """
    tmp = np.exp(array * theta)
    tmp /= np.sum(tmp)
    return tmp


def score_learning(episode_rewards, episode_losses, episode_fails):
    """
    compute a score for a learning process of an agent
    used for optimization tasks
    :param episode_rewards: rewards received during the training
    :param episode_losses: loss of the trained agents during the learning process
    :param episode_fails: ratio of invalid selected actions
    :return: three measures for the quality of the training
    """
    length = len(episode_rewards)
    weights = [x / length for x in range(length)]
    return sum(np.multiply(episode_rewards, weights)), sum(np.multiply(episode_losses, weights)), \
           sum(np.multiply(episode_fails, weights))


def lambda_rewards(rewards, gamma, lamb):
    """
    Transform returns of an episode of training into lambda-returns
    :param rewards: rewards got in the episode
    :param gamma: discount factor
    :param lamb: lambda hyperparameter to control fading away
    :return: rewards as lambda-returns
    """
    lambda_returns = []
    actual_return = 0
    for reward in rewards:
        actual_return = actual_return * gamma * lamb + reward
        lambda_returns.append(actual_return)
    return lambda_returns


def read_best_file(path="../best_benches.json"):
    """
    Read in the json file with the best alignments according to the different optimization scores
    and alignment methods (progressive alignment or iterative refinement) performed
    :param path: path to the location of the best results on the benchmarks
    :return: - the complete dictionary from the json file and
             - the filepath
    """
    current = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.isfile(current):
        with open(current, "r") as json_file:
            try:
                best = json.load(json_file)
            except JSONDecodeError:
                best = {"SP": {"Progressive": {}, "Refinement": {}}, "CS": {"Progressive": {}, "Refinement": {}}}
    else:
        best = {"SP": {"Progressive": {}, "Refinement": {}}, "CS": {"Progressive": {}, "Refinement": {}}}
    return best, current


def merge_best(computed, actual):
    """
    Merge the newly computed best-marks with the ones in the file
    in case of updated file best-marks from concurrent executions
    :param computed: newly computed best-marks
    :param actual: actual saved best-marks
    :return: merged, best of both inputs
    """
    output = {"SP": {"Progressive": {}, "Refinement": {}}, "CS": {"Progressive": {}, "Refinement": {}}}
    for i, score in enumerate(["SP", "CS"]):
        for mode in ["Progressive", "Refinement"]:
            for benchmark in set().union(*(ds.keys() for ds in [computed[score][mode], actual[score][mode]])):
                computed_score = float(computed[score][mode].get(benchmark, {"Score": "(-1, -1)"})["Score"].
                                       replace("(", "").replace(")", "").strip().split(",")[i])
                actual_score = float(actual[score][mode].get(benchmark, {"Score": "(-1, -1)"})["Score"].
                                     replace("(", "").replace(")", "").strip().split(",")[i])
                if computed_score > actual_score:
                    output[score][mode][benchmark] = computed[score][mode][benchmark]
                else:
                    output[score][mode][benchmark] = actual[score][mode][benchmark]
    return output


def write_best(best, current):
    """
    Write the collection of the actual best results on the benchmarks to a json-coded file
    :param best: best results on benchmarks
    :param current: file to write in
    """
    # update the actual values with values computed while computing bests'-results
    file_best, _ = read_best_file()
    best = merge_best(best, file_best)

    # test the writing of the file to not destroy the results stored in the old file
    with open(current, "w") as json_file, open("./tmp.json", "w") as tmp_file:
        json.dump(best, tmp_file, indent=4)
        json.dump(best, json_file, indent=4)
    os.remove("./tmp.json")


def read_fasta_data(file_name, replace_gaps=True, dna=False):
    """
    read sequence data from a fasta-encoded file holding the sequences to align
    :param file_name: file in fasta-format
    :param replace_gaps: flag indicating whether occasional gaps in the sequence files should be replaced
    :param dna: FLag for explicit input of DNA sequences; if false,
        sequence type is selected according to case of letters (uppercase = protein sequence, lowercase = DNA sequence)
    :return: list of sequences stored in the file
    """
    file = open(file_name, 'r')
    sequence = ''
    sequences = []
    header = ''
    headers = []
    for line in file.readlines():
        # if headline of a sequence: add old sequence and reset sequence storage
        if '>' in line:
            if len(sequence) != 0:
                if replace_gaps:
                    sequence = sequence.replace('-', '')
                if dna:
                    sequence = sequence.lower()
                sequences.append(sequence)
                headers.append(header)
                sequence = ''
            header = line.strip()
        # else append the line to the actual sequence
        else:
            sequence += line.strip()
    if replace_gaps:
        sequence = sequence.replace('-', '')
    if dna:
        sequence = sequence.lower()
    sequences.append(sequence)
    headers.append(header)
    return sequences, headers


def get_sequences(count=0, length=0, different=False, file=None):
    """
    return sequences to align, the returned sequences are variable in length and number
    this method is deterministic and therefore not has to compute the sequences every time from random
    can be used as small problem instance to debug algorithms on
    :param count: number of sequences
    :param length: length of each sequence
    :param different: flag indicating guaranteed difference of the returned sequences
    :param file: file to read for the sequences, if given all other arguments are ignored and all sequences are returned
    :return: sequences forming an instance of the multiple sequence alignment problem
    """
    if file is not None:
        return read_fasta_data(file)

    # only select as many sequences as possible
    count = min(count, 25)
    output = []
    i = 0

    # loop over all generated sequences and return as many sequences as specified or available within the specification
    while i < len(sequences) and len(output) < count:
        if not different or not sequences[i][:length] in output:
            output.append(sequences[i][:length])
        i += 1
    return output


def remove_char(oldstr, i):
    """
    remove char from string at specific position
    :param oldstr: old string with char to remove
    :param i: index of char to remove
    :return: new char without specified char
    """
    return oldstr[:i] + oldstr[(i + 1):]


def replace_char(oldstr, i, char):
    """
    replace char in string at specific position
    :param oldstr: old string with char to replace
    :param i: index to replace at
    :param char: char to insert at position i
    :return: new char with replaced char
    """
    return oldstr[:i] + char + oldstr[(i + 1):]


def generate_random_sequences(count, length, prob=10):
    """
    generate a new random sequence of DNA as alternative to the above given, that has also been created with this method
    :param count: number of randomly generated sequences
    :param length: length of the randomly generated sequences
    :param prob: probability of deleting or replacing a base
    :return: sequences generated by random
    """
    bases = ["A", "C", "G", "T"]
    sequence = ""
    sequences = []
    output_seq = []
    # create base sequence of DNA
    for i in range(length):
        sequence += random.choice(bases)
    sequences.append(sequence)
    # create variations of the basic DNA by replacing 10% of the bases (equal choice out of all 4 bases)
    for i in range(1, count):
        rand_seq = sequence
        for j in range(len(sequence)):
            if random.randint(0, prob - 1) == 0:
                rand_seq = replace_char(rand_seq, j, random.choice(bases))
        sequences.append(rand_seq)
    # Delete (1/prob)% of the bases in the sequences
    for seq in sequences:
        for j in range(len(sequence)):
            if random.randint(0, prob - 1) == 0:
                seq = remove_char(seq, j)
        output_seq.append(seq)
    print('\n'.join(sequences))
    return sequences


def get_sequence_type(seqs):
    """
    Determine the sequence type of a list of sequences
    :param seqs: sequences to investigate
    :return: Type, either DNA or PEP
    """
    return "DNA" if seqs[0].islower() else "PEP"


def get_sequence_size(seqs):
    """
    Compute the size properties of the given sequences
    :param seqs: sequences to use for investigation
    :return: number and average length of the sequences
    """
    return [len(seqs), sum([len(seq) for seq in seqs]) // len(seqs)]


def read_matrix(file_path, sep=" "):
    """
    Read a peptide-scoring matrix to use in the alignment process
    :param file_path: file of the stored values with the according bases at the beginning of each line and above
    :param sep: separator use in the file (e.g. space, tab, comma,...)
    :return: dictionary containing the read scoring matrix
    """
    file = open(file_path, 'r')
    head_line = file.readline().strip()
    output = dict()

    # read the head
    names = list(filter(lambda x: len(x) > 0, head_line.split(sep)))

    # read the single lines of the matrix
    for line in file.readlines():
        line = list(filter(lambda x: len(x) > 0, line.strip().split(sep)))
        output[line[0]] = {key: '0' if key in other else line[i + 1] for i, key in enumerate(names + other)}

    # fill in missing alphabetical values
    for o in other:
        output[o] = {key: '0' for key in names + other}

    return output


def write_matrix(file_path, matrix, sep="\t"):
    """
    Write a matrix into a file
    :param file_path: file to store the scoring matrix in
    :param matrix: scoring matrix representation
    :param sep: separator to use in score-separation
    """
    # write the header
    file = open(file_path, 'w')
    file.write(F"{sep}A{sep}B{sep}C{sep}D{sep}E{sep}F{sep}G{sep}H{sep}I{sep}J{sep}K{sep}L{sep}M"
               F"{sep}N{sep}O{sep}P{sep}Q{sep}R{sep}S{sep}T{sep}U{sep}V{sep}W{sep}X{sep}Y{sep}Z\n")

    # write each line
    for key in sorted(matrix.keys()):
        file.write(key + sep)
        for i, key2 in enumerate(sorted(matrix[key].keys())):
            file.write(matrix[key][key2])
            if i != 25:
                file.write(sep)
        file.write("\n")


def output_learning(comparison, configurations, names, refinement=False):
    """
    Print the learning results in a pretty-table to the commandline
    If there are more than 5 agents tested every 5 agents a new table is added with another 5 agents and the benchmarks
    :param comparison: results of the different agents in a dictionary per benchmark
    :param configurations: configurations of agents that were used to generate the results
    :param names: names of the benchmarks used
    :param refinement: flag indicating outputing the results of refinement analysis
    """
    # setup the naming for the columns
    header = ['Datasets', 'Type', '#S', '|S|', 'RL', 'DRL', 'CLUSTALW', 'MAFFT', 'MUSCLE'] + \
             (["Start"] if refinement else [])
    config_names = [str(n + 1) + ": " + ['TABLE', 'VALUE', 'POLICY', 'ACTOR_CRITIC', 'MCTS',
                                         'ALPHA0'][config.id - TABLE_AGENT] for n, config in enumerate(configurations)]

    # setup the tables
    pt = [PrettyTable() for _ in range(math.ceil(len(configurations) / 5))]
    for c in range(0, len(configurations), 5):
        pt[c // 5].field_names = header + config_names[c:min(c + 5, len(configurations))]

    for i, (b_id, typ, size, results) in enumerate(comparison):
        for c in range(0, len(configurations) + (1 if refinement else 0), 5):
            bound = min(c + 5, len(configurations) + (1 if refinement else 0))

            # add the row to the according table
            pt[c // 5].add_row([names[b_id] if isinstance(b_id, int) else os.path.basename(b_id)] + [typ] + size +
                               [print_scores(results[agent]) for agent in REFERENCES] +
                               [print_scores(results.get(i + TABLE_AGENT, [0, 0])[:2]) for i in range(c, bound)])

            # add a second row containing the results of the agents from the final eval-run on the problem instance
            pt[c // 5].add_row([""] * 9 + [print_scores(results.get(i + TABLE_AGENT, [0, 0, 0, 0])[2:])
                                           for i in range(c, bound)])

    # print all tables to the commandline
    for p in pt:
        print(p)


def print_scores(scores):
    """
    Print tuples of SP- and Column-Score
    :param scores: input tuple of scores
    :return: adjusted string-version of the score
    """
    if isinstance(scores[1], float):
        return str(scores[0]) + " | " + str(round(scores[1], 2))
    if scores[0] == scores[1] == 0:
        return ""
    if isinstance(scores[0], str) or isinstance(scores, list):
        return "n.a."
    return "failed"


def fasta_output(sequence, line_length=80):
    return '\n'.join(sequence[i:i + line_length] for i in range(0, len(sequence), line_length))


def get_score(config):
    """
    Extract the scoring of the config and turn it into a string
    Used as key generator for HashMaps
    :param config: config to get string representation for
    :return: string representation of configs score
    """
    return "SP" if config.score == SP_SCORE else "CS"


def get_c_type(config):
    """
    Extract alignment setting of the config and turn it into a string
    Used as key generator for HashMaps
    :param config: config to get string representation for
    :return: string representation of configs alignment setting
    """
    return "Refinement" if config.refinement else "Progressive"


def get_from_config(value, config):
    """
    Extract a value from a HashMap that contains values for both types of scoring and both alignment settings
    :param value: value to extract from
    :param config: config to use for extraction
    :return: value extracted from the input HashMap
    """
    return value[get_score(config)][get_c_type(config)]


def scan_benchmark_databases(folders, filtering=lambda x: True):
    """
    Search for possible benchmarks in folders. The folders should contain the benchmarks as individual Fasta-Files
    prints the final results to the commandline
    :param folders: folders to search in
    :param filtering: filter method
    """
    for folder in folders:
        print(folder, "\nMean\tAvg\t\tRange\tStd\t\tCount\t\tFile")
        results = []
        summing = [0, 0, 0, 0, 0]
        for file in os.listdir(folder):
            if os.path.isfile(folder + '/' + file) and \
                    (file.lower().endswith('.fa') or file.lower().endswith('.fasta') or file.lower().endswith('.tfa')):
                # extract some measures over the sequences in the file
                seqs, _ = read_fasta_data(folder + '/' + file)
                lengths = np.array([len(seq) for seq in seqs])
                len_rng = np.max(lengths) - np.min(lengths)
                len_mean = np.mean(lengths)
                len_avg = np.median(lengths)
                len_std = np.std(lengths)

                summing[0] += len_mean
                summing[1] += len_avg
                summing[2] += len_rng
                summing[3] += len_std
                summing[4] += len(seqs)

                # if they match the hard criteria, add them to the output
                if filtering((len_mean, len_avg, len_rng, len_std, len(seqs))):
                    results.append((len_avg, len_mean, len_rng, len_std, len(seqs), file))
        if len(results) != 0:
            summing = [s / len(results) for s in summing]
            results.append(tuple(summing + ["Average"]))
        for (len_avg, len_mean, len_rng, len_std, count, filename) in results:
            print("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%s" % (len_avg, len_mean, len_rng, len_std, count, filename))


@contextmanager
def suppress_stdout():
    """
    Source: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    Suppress the output of a part of the programm

    Use:
    with suppress_stdout():
        // Put code here
    continue with normal code and output
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def notify(message):
    """
    Notify via a telegram bot to the developer (therefore only works on devices that are connected to the internet)
    FOR SECURITY THIS METHOD HAS TO BE DELETED BEFORE RELEASE
    :param message: message to be sent to the
    :return: noting
    """
    # surround the messaging with try and error to catch case of being run without internet connection
    try:
        modified_message = message.replace("\n", "%0A").replace("\t", "")
        subprocess.call(['curl', '--data', 'parse_mode=HTML', '--data', 'chat_id=?????????', '--data',
                         F'text={modified_message}', '--request', 'POST',
                         'https://api.telegram.org/bot??????????:???????????????????????????????????/sendMessage'],
                        stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
    except OSError:
        print("No message sent: Cannot allocate memory")
    except RuntimeError:
        print("RuntimeError for some reason")
