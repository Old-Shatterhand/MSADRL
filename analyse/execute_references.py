import argparse
import os
import sys
import time
from shutil import copyfile

import pandas as pd

from utils.alignment import compute_similarity_matrix
from utils.profile import Profile

"""
Commandline arguments for running the reference algorithms
"""
run_clustal = "~/thesis/clustal/clustalo -i ~/input_seqs.fa -o ~/output_seqs.fa"
run_mafft = "~/thesis/mafft/bin/mafft --quiet ~/input_seqs.fa > ~/output_seqs.fa"
run_muscle = "~/thesis/muscle/muscle3.8.31_i86linux64 -in ~/input_seqs.fa -out ~/output_seqs.fa -quiet"
run_qscore = "~/apps/qscore/qscore -test ~/output_seqs.fa -ref ~/ref_seqs.fa > ~/log.txt"

run_clustal_tree = "~/thesis/clustal/clustalo -i ~/input_seqs.fa -o ~/output_seqs.fa --force --guidetree-out="
run_mafft_tree = "~/thesis/mafft/bin/mafft --quiet --treeout ~/input_seqs.fa > ~/output_seqs.fa"
run_muscle_tree = "~/thesis/muscle/muscle3.8.31_i86linux64 -in ~/input_seqs.fa -out ~/output_seqs.fa -quiet -tree2 "


def get_sequence_size(seqs):
    """
    Compute the size properties of the given sequences
    :param seqs: sequences to use for investigation
    :return: number and average length of the sequences
    """
    return [len(seqs), sum([len(seqs[seq]) for seq in seqs]) // len(seqs)]


def write_fasta_seqs(seqs, file_name="~/input_seqs.fa"):
    """
    Write sequences to output file give as filename
    :param seqs: sequences to write to file
    :param file_name: filepath to write to
    """
    file = open(os.path.expanduser(file_name), "w")
    for header in seqs.keys():
        file.write(">")
        file.write(header.strip())
        file.write("\n")
        file.write(seqs[header])
        file.write("\n\n")


def read_fasta_seqs(file_name="~/output_seqs.fa"):
    """
    read sequence data from a fasta-encoded file holding the sequences to align
    :param file_name: filepath to file to read sequences in fasta format from
    :return: list of sequences stored in the file
    """
    file = open(os.path.expanduser(file_name), "r")
    sequence = ''
    sequences = {}
    header = ''
    for line in file.readlines():
        # if headline of a sequence: add old sequence and reset sequence storage
        if '>' in line:
            if len(sequence) != 0 and len(header) != 0:
                sequences[header] = sequence
                sequence = ''
            header = line[1:].strip()
        # else append the line to the actual sequence
        else:
            sequence += line.strip()
    sequences[header] = sequence
    return sequences


def read_msf(file_name):
    """
    Read optimal handcrafted alignments from msa-formatted sequences
    :param file_name: filename to read alignment from
    :return: sequences in optimal alignment as dictionary
    """
    file = open(file_name, "r")
    read = False
    sequences = {}
    for line in file.readlines():
        if line[0:2] == "//":
            read = True
        if read and len(line) > 10:
            header = line[:12].strip()
            if header not in sequences:
                sequences[header] = ""
            sequences[header] += line[12:].strip().replace(" ", "").replace(".", "-")
    return sequences


def score_alignment():
    """
    Compute Q and TC score of alignments stored files in base-directory
    :return: scores computed by qscore as float
    """
    os.system(run_qscore)
    score_line = open(os.path.expanduser("~/log.txt"), "r").readline()
    os.remove(os.path.expanduser("~/log.txt"))
    print(score_line)
    parts = score_line.split(";")[2:]
    return float(parts[0].split("=")[1]), float(parts[1].split("=")[1])


def run_balibase(balibase_dir, code, table, results_dir):
    """
    Compute the performance of the three different reference tools (CLUSTAL, MAFFT, MUSCLE)
    :param balibase_dir: base-directory of the BAliBASE benchmark directory
    :param code: file-code from file to test
    :param table: table to store alignment results in
    """
    codes = ["RV20/BB20001", "RV20/BB20020", "RV40/BB40010", "RV40/BB40014", "RV40/BB40018", "RV50/BB50004"]
    if code not in codes:
        return
    print(code)

    # read the reference alignment and write it to the output
    base = os.path.join(balibase_dir, code)
    optimal = base + ".msf"
    input_file = base + ".tfa"
    input_seqs, _ = read_fasta_seqs(input_file)
    optimal_seqs = read_msf(optimal)
    ref_sp, ref_cs, _, _ = Profile(list(optimal_seqs.values())).score()
    write_fasta_seqs(optimal_seqs, "~/ref_seqs.fa")
    write_fasta_seqs(input_seqs)

    # perform the alignment using CLUSTAL
    start = time.time()
    os.system(run_clustal)
    clustal_time = time.time() - start
    clustal_sp, clustal_cs, _, _ = Profile(list(read_fasta_seqs().values())).score()
    clustal_q, clustal_ts = score_alignment()
    if results_dir is not None:
        os.rename(os.path.expanduser("~/output_seqs.fa"),
                  os.path.join(results_dir, "CLUSTAL_" + code.split("/")[1] + ".fa"))
    else:
        os.remove(os.path.expanduser("~/output_seqs.fa"))
    print("CLUSTAL:", clustal_q, ",\t", clustal_ts, ",\t", clustal_cs, ",\t", clustal_sp)

    # perform the alignment using MAFFT
    start = time.time()
    os.system(run_mafft)
    mafft_time = time.time() - start
    mafft_sp, mafft_cs, _, _ = Profile(list(read_fasta_seqs().values())).score()
    mafft_q, mafft_ts = score_alignment()
    if results_dir is not None:
        os.rename(os.path.expanduser("~/output_seqs.fa"),
                  os.path.join(results_dir, "MAFFT_" + code.split("/")[1] + ".fa"))
    else:
        os.remove(os.path.expanduser("~/output_seqs.fa"))
    print("MAFFT:\t", mafft_q, ",\t", mafft_ts, ",\t", mafft_cs, ",\t", mafft_sp)

    # perform the alignment using MUSCLE
    start = time.time()
    os.system(run_muscle)
    muscle_time = time.time() - start
    muscle_sp, muscle_cs, _, _ = Profile(list(read_fasta_seqs().values())).score()
    muscle_q, muscle_ts = score_alignment()
    if results_dir is not None:
        os.rename(os.path.expanduser("~/output_seqs.fa"),
                  os.path.join(results_dir, "MUSCLE_" + code.split("/")[1] + ".fa"))
    else:
        os.remove(os.path.expanduser("~/output_seqs.fa"))
    print("MUSCLE:\t", muscle_q, ",\t", muscle_ts, ",\t", muscle_cs, ",\t", muscle_sp)

    # delete not necessary files and save the results in a table
    os.remove(os.path.expanduser("~/input_seqs.fa"))
    os.remove(os.path.expanduser("~/ref_seqs.fa"))
    if code not in table["name"]:
        table.loc[len(table)] = [code, len(optimal_seqs), get_sequence_size(optimal_seqs), ref_cs, ref_sp,
                                 compute_identity(balibase_dir, code), clustal_q, clustal_ts, clustal_cs, clustal_sp,
                                 clustal_time, mafft_q, mafft_ts, mafft_cs, mafft_sp, mafft_time, muscle_q, muscle_ts,
                                 muscle_cs, muscle_sp, muscle_time]


def create_trees_only(balibase_dir, code, tree_path):
    codes = ["RV20/BB20001", "RV20/BB20020", "RV40/BB40010", "RV40/BB40014", "RV40/BB40018", "RV50/BB50004"]
    if code[2] != "1" and code not in codes:
        print("Return:", code)
        return

    print("Run", code)

    write_fasta_seqs(read_fasta_seqs(os.path.join(balibase_dir, code + ".tfa")))
    clustal_tree_file = os.path.join(tree_path, "CLUSTAL_" + code.split("/")[1] + "_tree.txt")
    mafft_tree_file = os.path.join(tree_path, "MAFFT_" + code.split("/")[1] + "_tree.txt")
    muscle_tree_file = os.path.join(tree_path, "MUSCLE_" + code.split("/")[1] + "_tree.txt")
    os.system(run_clustal_tree + clustal_tree_file)
    os.system(run_mafft_tree)
    os.rename(os.path.expanduser("~/input_seqs.fa.tree"), mafft_tree_file)
    os.system(run_muscle_tree + muscle_tree_file)


def run_benchmark(balibase_dir, table=None, result_dir=None, tree_dir=None):
    """
    Run the benchmarks starting from the base directory of BAliBASE by listing the relevant folders and files
    :param balibase_dir: base-directory of balibase
    :param table: table to store the alignments in
    :param result_dir: folder to store the alignments in
    """
    for folder in sorted(os.listdir(balibase_dir)):
        if os.path.isdir(os.path.join(balibase_dir, folder)) and "bali_score_src" not in folder:
            codes = set()
            for file in sorted([f for f in os.listdir(os.path.join(balibase_dir, folder)) if
                                os.path.isfile(os.path.join(balibase_dir, folder, f))]):
                name = file.split(".")[0]
                if name not in codes and file[2] != "S":
                    if table is not None and result_dir is not None:
                        run_balibase(balibase_dir, os.path.join(folder, name), table, result_dir)
                    elif tree_dir is not None:
                        create_trees_only(balibase_dir, os.path.join(folder, name), tree_dir)
                    codes.add(name)


def read_drl_results(balibase_dir, results_dir, agent_name, df):
    """
    Insert results into the dataframe from the comparison of the reference tools
    :param balibase_dir: directory of the balibase database
    :param results_dir: directory of the result files from this tool
    :param agent_name: agent prefix inserted into the result files
    :param df: dataframe from the comparison
    :return: extended dataframe
    """
    # if necessary, insert the according columns
    if agent_name + "_q" not in df.columns:
        df[agent_name + "_q"] = 0
        df[agent_name + "_tc"] = 0
        df[agent_name + "_cs"] = 0
        df[agent_name + "_sp"] = 0
        df[agent_name + "_time"] = 0

    # iterate on all the files
    for file_name in sorted([f for f in os.listdir(results_dir) if agent_name in f]):
        print(file_name)
        # preprocess the analysis
        key = "RV" + file_name[-9:-7] + "/" + file_name[-11:-4]
        optimal = os.path.join(balibase_dir, key + ".msf")
        write_fasta_seqs(read_msf(optimal), "~/ref_seqs.fa")
        copyfile(os.path.join(results_dir, file_name), os.path.expanduser("~/output_seqs.fa"))

        # score the alignment
        sp, cs, _, _ = Profile(list(read_fasta_seqs().values())).score()
        q, tc = score_alignment()

        # insert the data into the  dataframe
        df.loc[df.name == key, agent_name + "_q"] = q
        df.loc[df.name == key, agent_name + "_tc"] = tc
        df.loc[df.name == key, agent_name + "_cs"] = cs
        df.loc[df.name == key, agent_name + "_sp"] = sp

    os.remove(os.path.expanduser("~/output_seqs.fa"))
    os.remove(os.path.expanduser("~/ref_seqs.fa"))

    return df


def insert_data_file(data_file, balibase_dir, results_dir, df):
    print("Reached function")
    for line in open(data_file, 'r'):
        data = line.split("\t")
        name = data[0] + "_" + data[3]
        benchmark = data[1]
        runtime = data[4]
        SP = data[5]
        C = data[6]
        permutation = data[7]
        counts = data[8]

        if name + "_q" not in df.columns:
            df[name + "_q"] = 0
            df[name + "_tc"] = 0
            df[name + "_cs"] = 0
            df[name + "_sp"] = 0
            df[name + "_time"] = 0
            df[name + "_permutation"] = 0
            df[name + "_counts"] = 0

        key = "RV" + benchmark[2:4] + "/" + benchmark
        write_fasta_seqs(read_msf(os.path.join(balibase_dir, key + ".msf")), "~/ref_seqs.fa")
        print(("0_Policy" if name == "Policy" else "1_ActorCritic") + "_SP_P_" + benchmark + ".tfa")
        copyfile(os.path.join(results_dir, ("0_Policy" if name == "Policy" else "1_ActorCritic") + "_SP_P_" +
                              benchmark + ".tfa"), os.path.expanduser("~/output_seqs.fa"))

        # score the alignment
        q, tc = score_alignment()
        print(name, "on", benchmark, ":", q, ", ", tc, "(", permutation, ")")

        # insert the data into the  dataframe
        df.loc[df.name == key, name + "_q"] = q
        df.loc[df.name == key, name + "_tc"] = tc
        df.loc[df.name == key, name + "_cs"] = C
        df.loc[df.name == key, name + "_sp"] = SP
        df.loc[df.name == key, name + "_time"] = runtime
        df.loc[df.name == key, name + "_permutation"] = permutation
        df.loc[df.name == key, name + "_counts"] = counts

        os.remove(os.path.expanduser("~/output_seqs.fa"))

    os.remove(os.path.expanduser("~/ref_seqs.fa"))

    return df


def compute_identity(balibase_dir, file_code):
    sequences = read_fasta_seqs(os.path.join(balibase_dir, file_code + ".tfa"))
    sequences = list(sequences.values())
    if len(sequences) > 10 or sum([len(seq) for seq in sequences]) / len(sequences) > 500:
        return 0
    print("\rSimilarity for", file_code, end="")
    matrix = compute_similarity_matrix(sequences, [""] * len(sequences), True)
    summing, count = 0, 0
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            summing += float(matrix[i][j + 1])
            count += 1
    return round(summing / count, 2)


if __name__ == '__main__':
    # Parse the arguments
    arg_parser = argparse.ArgumentParser(description="Execute the reference tools of CLUSTAL, MUSCLE and MAFFT on"
                                                     "the BAliBASE and store the results in a table. The paths to the "
                                                     "files are hard-coded in the script. Additionally, results "
                                                     "computed with this algorithm can be inserted in the table.")
    arg_parser.add_argument("-b", "--balibase", dest="balibase", nargs=1, type=str,
                            help="Directory of the BAliBASE to read the sequences from.")
    arg_parser.add_argument("-c", "--comparison", dest="comparison", nargs=1, type=str,
                            help="File the comparison is stored in.")
    arg_parser.add_argument("-r", "--results", dest="results", nargs=1, type=str,
                            help="Folder of the result files from an execution of this script.")
    arg_parser.add_argument("-n", "--name", dest="name", nargs=1, type=str,
                            help="Name of the Agent used for the results to insert.")
    arg_parser.add_argument("-d", "--data", dest="data", nargs=1, type=str,
                            help="Datafile, that was created when executing the main program.")
    arg_parser.add_argument("-t", "--tree", dest="tree", nargs=1, type=str,
                            help="Folder to store the trees in. This option disables any other analysis.")

    args = arg_parser.parse_args(sys.argv[1:])

    # and start the program according to the specified parameters
    if args.balibase is not None and os.path.isdir(args.balibase[0]):
        if args.comparison is not None and os.path.isfile(args.comparison[0]) and \
                args.results is not None and os.path.isdir(args.results[0]):
            if args.name is not None:
                print("Insert own results")
                df = read_drl_results(args.balibase[0], args.results[0], args.name[0], pd.read_csv(args.comparison[0]))
                df.to_csv(args.comparison[0], index=False)
            elif args.data is not None:
                print("Insert from data-file")
                df = insert_data_file(args.data[0], args.balibase[0], args.results[0], pd.read_csv(args.comparison[0]))
                df.to_csv(args.comparison[0], index=False)
            else:
                print("Analyse references")
                # create dataframe and perform the analysis
                df = pd.DataFrame([], columns=["name", "nseqs", "lseqs", "ref_cs", "ref_sp", "avg_ident",
                                               "clustal_q", "clustal_ts", "clustal_cs", "clustal_sp", "clustal_time",
                                               "mafft_q", "mafft_ts", "mafft_cs", "mafft_sp", "mafft_time",
                                               "muscle_q", "muscle_ts", "muscle_cs", "muscle_sp", "muscle_time"])
                run_benchmark(args.balibase[0], df, args.results[0])
                df.to_csv(args.comparison[0], index=False)
        elif args.tree is not None and os.path.isdir(args.tree[0]):
            run_benchmark(args.balibase[0], tree_dir=args.tree[0])
    else:
        arg_parser.print_help()
