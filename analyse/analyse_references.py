import argparse
import sys

from prettytable import PrettyTable

from optimizer import read_best_file

dna_benchmarks = ['Hepatitis-C-Virus', 'Papio Anubis', 'Oxbench 469', 'Oxbench 429', 'LGM-Dataset', 'RLO-Dataset',
                  'Dataset 1', 'Oxbench 414', 'Oxbench 415']
protein_benchmarks = ['Oxbench 433', 'Oxbench 641t2', 'Oxbench 34', 'Oxbench 620']


def output(data):
    """
    Print the analysed data as PrettyTables to the console. The output distinguishes sequence types (DNA and Proteins)
    and the used general algorithms, not the specific configurations
    :param data: analysed data to output
    """
    pt = PrettyTable()
    pt.field_names = ["Algorithm", "DNA", "Protein"]
    for key in set().union(*(ds.keys() for ds in [data["DNA"], data["Protein"]])):
        pt.add_row([key, str(data["DNA"].get(key, 0)), str(data["Protein"].get(key, 0))])
    print(pt)


def analyse(data):
    """
    Analyse the input data in that way, that it count which basic algorithm found the optimal alignments for each
    benchmark differentiated into DNA and Protein Benchmarks
    :param data: dictionary of the benchmarks (so the score to optimize and the type of alignment is fixed)
    :return: dictionary containing the counts for all agents that have at least one optimal alignment in two dicts,
    one for proteins and one for DNA
    """
    output = {"DNA": {}, "Protein": {}}

    # count the DNA optimal alignment algorithms
    for i in [data[benchmark]["Configuration"]["Name"] for benchmark in data.keys() if
              benchmark in dna_benchmarks]:
        output["DNA"][i] = output["DNA"].get(i, 0) + 1

    # count the Protein optimal alignment algorithms
    for i in [data[benchmark]["Configuration"]["Name"] for benchmark in data.keys() if
              benchmark in protein_benchmarks]:
        output["Protein"][i] = output["Protein"].get(i, 0) + 1

    return output


def output_cumulative(data_sets):
    """
    Compute the cumulative statistics for the list or tuple of input data based on the analysis results
    Finally calls output with the result to print the resulting table
    :param data_sets: analysed datasets to compute the cumulative statistics for
    """
    cumulative = {"DNA": {}, "Protein": {}}

    # merge the DNA statistics
    all_dna_keys = set().union(*(ds["DNA"].keys() for ds in data_sets))
    cumulative["DNA"] = {key: sum([ds["DNA"].get(key, 0) for ds in data_sets]) for key in all_dna_keys}

    # merge the Protein statistics
    all_protein_keys = set().union(*(ds["Protein"].keys() for ds in data_sets))
    cumulative["Protein"] = {key: sum([ds["Protein"].get(key, 0) for ds in data_sets]) for key in all_protein_keys}

    output(cumulative)


def output_all(data):
    """
    Compute and print the statistics for all configurations of learning and the accumulated form
    :param data: input data directly from the json file
    """
    sp_ana = analyse(data["SP"]["Progressive"])
    sr_ana = analyse(data["SP"]["Refinement"])
    cp_ana = analyse(data["CS"]["Progressive"])
    cr_ana = analyse(data["CS"]["Refinement"])

    print("All optimization statistics of SP- and C-Score optimization for progressive and refinement tasks")
    output_cumulative((sp_ana, sr_ana, cp_ana, cr_ana))

    print("Best Progressive SP-Score optimizing agents")
    output(sp_ana)

    print("Best Refinement SP-Score optimizing agents")
    output(sr_ana)

    print("Best Progressive C-Score optimizing agents")
    output(cp_ana)

    print("Best Refinement C-Score optimizing agents")
    output(cr_ana)


def output_one(data, title):
    """
    Conpute and print the statistics only for one configuration of learning
    :param data: input data from filtered json-data
    :param title: title to write above the table
    """
    print(title)
    output(data)


def output_two(data1, data2, all_title, title1, title2):
    """
    Compute and print the statistics for two configurations of learning
    :param data1: data of first configuration
    :param data2: data of second configuration
    :param all_title: title of complete analysis
    :param title1: title of first configuration
    :param title2: title of second configuration
    """
    print(all_title, end="\n\n")
    output_cumulative((data1, data2))

    output_one(data1, title1)
    print()

    output_one(data2, title2)


def eval_query(data, total=False, sp=False, cs=False, prog=False, refine=False):
    """
    Evaluate the query to the analysis tool to print the specified configuration(s)
    sp and cs can be conjunct, prog and refine con be conjunct, sp, cs and prog, refine disjointing each other
    :param data: data (directly from json-file) to select from
    :param total: flag to print the complete data (override all other flags)
    :param sp: flag to analyse sp score
    :param cs: flag to analyse c score
    :param prog: flag to analyse the progressive alignments
    :param refine: flag to analyse the iterative refinements
    """
    if total or (sp and cs and prog and refine):
        output_all(data)
    elif sp and cs:
        if prog:
            output_two(data["CS"]["Progressive"], data["SP"]["Progressive"],
                       "Best Progressive SP- and C-Score optimizing agents",
                       "Best Progressive Column-Score optimizing agents", "Best Progressive SP-Score optimizing agents")
        elif refine:
            output_two(data["CS"]["Refinement"], data["SP"]["Refinement"],
                       "Best Refinement SP- and C-Score optimizing agents", "Best Refinement C-Score optimizing agents",
                       "Best Refinement SP-Score optimizing agents")
        else:
            print("Invalid query")
    elif sp:
        if prog and refine:
            output_two(data["SP"]["Progressive"], data["SP"]["Refinement"],
                       "Best Progressive and Refinement SP-Score optimizing agents",
                       "Best Progressive SP-Score optimizing agents", "Best Refinement SP-Score optimizing agents")
        elif prog:
            output_one(data["SP"]["Progressive"], "Best Progressive SP-Score optimizing agents")
        elif refine:
            output_one(data["SP"]["Refinement"], "Best Refinement SP-Score optimizing agents")
        else:
            print("Invalid query")
    elif cs:
        if prog and refine:
            output_two(data["CS"]["Progressive"], data["CS"]["Refinement"],
                       "Best Progressive and Refinement C-Score optimizing agents",
                       "Best Progressive C-Score optimizing agents", "Best Refinement C-Score optimizing agents")
        elif prog:
            output_one(data["CS"]["Progressive"], "Best Progressive C-Score optimizing agents")
        elif refine:
            output_one(data["CS"]["Refinement"], "Best Refinement C-Score optimizing agents")
        else:
            print("Invalid query")
    else:
        print("Invalid query")


if __name__ == '__main__':
    best, _ = read_best_file()
    arg_parser = argparse.ArgumentParser(description="Analysis tool to visualize the data from the json-file "
                                                     "containing the best alignments for each way of alignment and "
                                                     "optimization score by counting the algorithms that optimize the "
                                                     "benchmarks in the individual settings.")
    arg_parser.add_argument("-a", "--all", dest="All", default=False, action="store_true",
                            help="Print statistics of the agents in all optimization tasks")
    arg_parser.add_argument("-s", "--sp", dest="SP", default=False, action="store_true",
                            help="Print statistics of the agents in sp-score optimization tasks")
    arg_parser.add_argument("-c", "--cs", dest="CS", default=False, action="store_true",
                            help="Print statistics of the agents in c-score optimization tasks")
    arg_parser.add_argument("-p", "--prog", dest="Prog", default=False, action="store_true",
                            help="Print statistics of the agents in progressive alignment optimization tasks")
    arg_parser.add_argument("-r", "--refine", dest="Refine", default=False, action="store_true",
                            help="Print statistics of the agents in iterative refinement optimization tasks")
    result = arg_parser.parse_args(sys.argv[1:])

    eval_query(best, result.All, result.SP, result.CS, result.Prog, result.Refine)
