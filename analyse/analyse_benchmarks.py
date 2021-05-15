import argparse
import os
import sys

import pandas as pd
import prettytable as pt

# number of position behind the comma to print in the output of the table
precision = 2


class Results:
    def __init__(self):
        """
        A class to hold the results of the comparison in a comprimized way
        """
        self.results = {key: {"Count": 0, "Q": {"Won": 0, "Draw": 0, "C": 0, "M": 0, "F": 0, "D": {}},
                              "TC": {"Won": 0, "Draw": 0, "C": 0, "M": 0, "F": 0, "D": {}},
                              "SP": {"Won": 0, "Draw": 0, "C": 0, "M": 0, "F": 0, "D": {}},
                              "C": {"Won": 0, "Draw": 0, "C": 0, "M": 0, "F": 0, "D": {}}} for key in
                        ["RV11_100", "RV11_250", "RV11_500", "RV12_100", "RV12_250", "RV12_500", "RV2_100", "RV2_250",
                         "RV2_500", "RV3_100", "RV3_250", "RV3_500", "RV4", "RV5"]}

    def set(self, group, row, key):
        """
        Add the data of a row to the according fields in the internal representation
        :param group: groups name of the row
        :param row: row to be inserted
        :param key: name of the agent used for the analysis
        """
        self.results[group]["Count"] += 1
        for score, score_key in [("Q", "q"), ("TC", "tc"), ("SP", "sp"), ("C", "cs")]:
            best_ref = float("-inf")
            worst_ref = float("inf")
            best_own = float("-inf")
            for algo, algo_key in [("C", "clustal"), ("M", "mafft"), ("F", "mafft"), ("D", key)]:
                if algo == "D":
                    for inner_key in algo_key:
                        if inner_key not in self.results[group][score][algo]:
                            self.results[group][score][algo][inner_key] = 0
                        self.results[group][score][algo][inner_key] += row[inner_key + "_" + score_key]
                        best_own = max(best_own, row[inner_key + "_" + score_key])
                else:
                    self.results[group][score][algo] += row[algo_key + "_" + score_key]
                    best_ref = max(best_ref, row[algo_key + "_" + score_key])
                    worst_ref = min(worst_ref, row[algo_key + "_" + score_key])
            if best_own > best_ref:
                self.results[group][score]["Won"] += 1
            if best_own > worst_ref:
                self.results[group][score]["Draw"] += 1

    def finish(self):
        """
        Finish the insertion by dividing all non-null positions by the count of added values
        """
        for key in self.results.keys():
            for score, score_key in [("Q", "q"), ("TC", "tc"), ("SP", "sp"), ("C", "cs")]:
                for algo, algo_key in [("C", "clustal"), ("M", "mafft"), ("F", "mafft"), ("D", key)]:
                    if self.results[key]["Count"] != 0:
                        if algo == "D":
                            self.results[key][score][algo] = {inner_key: self.results[key][score][algo][inner_key] /
                                                                         self.results[key]["Count"] for inner_key in
                                                              self.results[key][score][algo].keys()}
                        else:
                            self.results[key][score][algo] /= self.results[key]["Count"]

    def to_table(self, keys):
        """
        Output the data in four tables for each optimization score
        """
        for score in ["Q", "TC", "SP", "C"]:
            table = pt.PrettyTable(["Group", "Count", "WRatio", "DRatio", "CLUSTAL", "MUSCLE", "MAFFT"] + keys)
            table.title = "Scoring: " + score
            for key in ["RV11_100", "RV11_250", "RV11_500", "RV12_100", "RV12_250", "RV12_500", "RV2_100", "RV2_250",
                        "RV2_500", "RV3_100", "RV3_250", "RV3_500", "RV4", "RV5"]:
                # only insert the data if it is relevant and not just zero
                if self.results[key]["Count"] != 0:
                    table.add_row([key, self.results[key]["Count"],
                                   round(self.results[key][score]["Won"] / self.results[key]["Count"], precision),
                                   round(self.results[key][score]["Draw"] / self.results[key]["Count"], precision),
                                   round(self.results[key][score]["C"], precision),
                                   round(self.results[key][score]["M"], precision),
                                   round(self.results[key][score]["F"], precision)] + [
                                      round(self.results[key][score]["D"][inner_key], precision) for inner_key in
                                      self.results[key][score]["D"].keys()])
            print(table)


def get_group(name, seq_count):
    """
    Compute the name of the group in BAliBASE according to the name of the file and the sequence count
    :param name: name of the file
    :param seq_count: number of sequences in the file
    :return: name of the group used in this analysis
    """
    prefix = name.split("/")[0].replace("0", "")

    # if the prefix of the sequence is already the groups' name, return it (only for RV4 and 5)
    if prefix == "RV4" or prefix == "RV5":
        return prefix

    # for sets 1 and 2 distinguish the number of sequences in the file
    if seq_count < 100:
        return prefix + "_100"
    if seq_count < 500:
        return prefix + "_250"
    return prefix + "_500"


def analyse_balibase(df, key):
    """
    Analyse the comparison for the balibase entries
    :param df: dataframe according to the comparison file
    :param key: name of the agent to use in the comparison
    """
    results = Results()

    # insert the data into the structure to print and analyse
    for index, row in df.iterrows():
        if sum([sum([row[inner_key + score] for inner_key in key]) for score in ["_sp", "_cs"]]) != 0:
#            print(row["name"], "|", row["lseqs"], ":\t", get_group(row["name"], row["lseqs"]) + ",\t" + str(row["avg_ident"]))
            results.set(get_group(row["name"], row["lseqs"]), row, key)

    # finish the analysis and print
    results.finish()
    results.to_table(key)


if __name__ == "__main__":
    # parse the arguments
    arg_parser = argparse.ArgumentParser(description="Analyse the comparison from the execute_reference script.")
    arg_parser.add_argument("-c", "--comparison", dest="comparison", nargs=1, type=str,
                            help="File the comparison is stored in.")
    arg_parser.add_argument("-n", "--name", dest="name", nargs='+', type=str,
                            help="Name of the Agent used for the results to insert.")
    args = arg_parser.parse_args(sys.argv[1:])

    # and start the program according to the input
    if args.comparison[0] is not None and os.path.isfile(args.comparison[0]) and args.name[0] is not None:
        analyse_balibase(pd.read_csv(args.comparison[0], header=0, index_col=None), args.name)
    else:
        arg_parser.print_help()
