import argparse
import json
import os
import sys
import time

import numpy as np
import warnings

import torch

from networks.actorcritic_networks import AC_Network
from networks.alphazero_networks import A0_Network
from networks.ffnn_networks import ValueFFNN
from networks.reinforce_networks import REINFORCENetwork
from optimizer import run_multiple_agents
from utils.alignment import multiprocessing_brute_force, similarity_matrix, \
    center_star, brute_force_alignment
from utils.configurations import TableAgentTrainingConfiguration as TATC, ValueAgentTrainingConfiguration as VATC, \
    PolicyAgentTrainingConfiguration as PATC, ActorCriticAgentTrainingConfiguration as ACATC, \
    MCTSTrainingConfiguration as MCTSTC, AlphaZeroTrainingConfiguration as A0TC, from_dict
from utils.constants import benchs, SP_SCORE, C_SCORE, seq_files
from utils.utils import notify, read_fasta_data

"""
Handling the commandline interface and algorithm starting of the entire work
"""


def join_paths(p1, p2):
    """
    Join two paths, one, p1, relative to the current working directory (cwd) and  the second, p2, relative to the first
    :param p1: path relative to the cwd (e.g. file-path of the json file)
    :param p2: path relative to p1 (e.g. file-path from json to benchmarks
    :return: file absolute path to the second file
    """
    # split paths into different folders they traverse
    abs_p1 = os.path.abspath(p1)
    rel_p2 = p2.replace("\\", os.path.sep).replace("/", os.path.sep)
    p1_parts = list(filter(lambda x: not (len(x) == 0 and x != '.'), abs_p1.split(os.path.sep)))
    p2_parts = list(filter(lambda x: not (len(x) == 0 and x != '.'), rel_p2.split(os.path.sep)))

    # delete file from the first path
    if os.path.isfile(abs_p1):
        p1_parts = p1_parts[:-1]

    # go up in the first path as long as the second path goes up
    while p2_parts[0] == "..":
        p1_parts = p1_parts[:-1]
        p2_parts = p2_parts[1:]

    return os.path.sep.join(p1_parts + p2_parts)


def join_config_files(configurations):
    """
    Merge multiple config files in following ways:
    - Merge the agents, i.e.: combine the different agent configurations and store equal configurations only once
    - Merge the benchmark assignments to fit the new list of agents
    :param configurations: configurations to merge
    :return: list of combined agent configurations and accordingly merged benchmark assignments
    """
    configs = []
    benchmarks = {}
    n = 0
    for agents, benchs in configurations:
        b, tmp = 0, {}

        # merge agents
        for i, agent in enumerate(agents):
            if agent in configs:
                tmp[i] = configs.index(agent)
                b += 1
            else:
                tmp[i] = n + i - b
                configs.append(agent)

        # merge benchmarks on the new agent indices
        for bench, agents in benchs.items():
            agents = list(map(lambda x: tmp[x], agents))
            blub = set(benchmarks.get(bench, []))
            blub.update(set(agents))
            benchmarks[bench] = list(blub)
        n = len(configs)
    return configs, benchmarks


def parse_config_file(file_path):
    """
    Parse a single config-file into the according agent configurations and the benchmarks
    :param file_path: file path of the configurations file
    :return: parsed agents and benchmarks assignments
    """
    with open(file_path) as json_file:
        data = json.load(json_file)
        agents = []
        benchmarks = {}

        # parse the agents
        for agent in data["Agents"]:
            agents.append(from_dict(agent))

        # parse the benchmarks
        for mark in data["Benchmarks"]:
            benchmarks[benchs.get(mark['Name'], os.path.join("/".join(file_path.split("/")[:-1]), mark['Name']))] = \
                [x for x in mark["IDs"]]
        return agents, benchmarks


def parse_arguments(args):
    """
    Parse the arguments if no config-file is given into the different configurations
    :param args: arguments to pass
    :return: from arguments resulting training-configuration
    """
    if args.Agent[0] == 'T':  # Tabular Agent
        return TATC(args.Games, 0, 0, args.Alpha, args.Gamma, args.Lambda, args.N, args.Optimize, args.Look,
                    args.Support, args.Progress, args.Graph, args.Notify, args.Refinement, "Table")
    elif args.Agent[0] == 'V':  # Function approximating Agent
        return VATC(args.Games, 0, 0, 0, 0, args.Alpha, args.Gamma, args.Lamb, args.N, args.Optimize, ValueFFNN,
                    args.Look, args.Support, args.Progress, args.Graph, args.Notify, args.Refinement, "Value")
    elif args.Agent[0] == 'P':  # Policy approximating Agent
        return PATC(args.Games, 0, 0, args.Alpha[0], args.Gamma[0], args.Alpha[1], args.Gamma[1], True, args.Optimize,
                    REINFORCENetwork, args.Look, args.Support, args.Progress, args.Graph, args.Notify, args.Refinement,
                    "Policy")
    elif args.Agent[0] == 'A':  # Actor-Critic Agent
        return ACATC(args.Games, 0, 0, args.Alpha[0], args.Gamma[0], args.Alpha[1], args.Gamma[1], args.Optimize,
                     AC_Network, args.Look, args.Support, args.Progress, args.Graph, args.Notify, args.Refinement,
                     "ActorCritic")
    elif args.Agent[0] == 'M':  # MCTS/UTC-Agent
        return MCTSTC(args.Simulations, args.Rollouts, args.C, args.Optimize, args.Progress, args.Notify,
                      args.Refinement, "MCTS")
    elif args.Agent[0] == '0':  # Alpha-Zero Agent
        return A0TC(args.Games, args.Simulations, args.Rollouts, args.C, args.Optimize, A0_Network, args.Progress,
                    args.Graph, args.Notify, args.Refinement, "Alpha")
    else:
        return None


def get_arg_parser():
    """
    Create the argument parser for the commandline arguments
    :return: return the argparser object to be used to parse the commandline arguments
    """
    arg_parser = argparse.ArgumentParser(description="Tool to align multiple biological sequences using techniques of "
                                                     "Deep Reinforcement Learning\nFor a detailed description of e.g."
                                                     "Which arguments can be provided for which agents and how many "
                                                     "are needed cen be shown up in the README.md of this program")
    arg_parser.add_argument("-a", "--agent", dest="Agent", default=None, nargs=1,
                            choices=['T', 'V', 'P', 'A', 'M', '0'],
                            help='Specify the agent to use to perform the alignments')
    arg_parser.add_argument("-g", "--games", dest="Games", type=int, default=0, nargs=1,
                            help='Number of games to simulate in the training')

    arg_parser.add_argument("-A", "--alpha", dest="Alpha", type=float, default=0.01, action='append', nargs='+',
                            help='learning-rate alpha, first value determines the rate for the value estimae of an '
                                 'agent. The second value sets the rate for the policy approximator.')
    arg_parser.add_argument("-G", "--gamma", dest="Gamma", type=float, default=0.99, action='append', nargs='+',
                            help='discount factor gamma, the first determines lambda for value estimators, the second '
                                 'for the policy approximator')
    arg_parser.add_argument("-L", "--lambda", dest="Lambda", type=float, default=1, action='append', nargs='+',
                            help='Lambda to use for the lambda-return based learning of the agent. The first determines'
                                 ' lambda for value estimators, the second for policy approximators')

    arg_parser.add_argument("-N", "--nstep", dest="N", type=int, default=0, nargs=1,
                            help='Number of steps to take into account in n-step learning. A value of -1 indicates the '
                                 'use Monte-Carlo-Returns whereas -2 indicates the use of Lambda-Returns.')

    arg_parser.add_argument("-B", "--baseline", dest="Baseline", default=False, action="store_true",
                            help='Flag for the usage of a value-function approximating baseline in REINFORCE Agents')
    arg_parser.add_argument("-S", "--simulations", dest="Simulations", type=int, default=0, nargs=1,
                            help='number of simulations (MCTS equivalent for games) to do before choosing an action')
    arg_parser.add_argument("-R", "--rollouts", dest="Rollouts", type=int, default=1, nargs=1,
                            help='number of rollouts per simulation')
    arg_parser.add_argument("-C", "--hyperc", dest="C", type=float, default=1, nargs=1,
                            help='exploitation-exploration balancing hyperparameter in MCTS')

    arg_parser.add_argument("-l", "--look-ahead", dest="Look", default=False, action='store_true',
                            help='Flag turning the look-ahead-search on')
    arg_parser.add_argument("-s", "--s-search", dest="Support", default=False, action='store_true',
                            help='Flag for the search support')

    arg_parser.add_argument("-O", "--optimize", dest="Optimize", default="SP", nargs=1, choices=['SP', 'C'],
                            help="Score to optimize the agent for")
    arg_parser.add_argument("-r", "--refine", dest="Refinement", default=False, action='store_true',
                            help='Specify whether the agent yould be run as refinement or progressive agent')
    arg_parser.add_argument("-F", "--brute-force", dest="BruteForce", default=False, action='store_true',
                            help='Flag to be set to perform brute-force alignment on the set benchmark. '
                                 'This flag requires -b to be additionally set.')

    arg_parser.add_argument("-p", "--progress", dest="Progress", default=False, action='store_true',
                            help='Printing the output of the training to the console')
    arg_parser.add_argument("-o", "--plot", dest="Graph", default=False, action='store_true',
                            help='creating a graph with curves for reward, loss, etc. after learning')
    arg_parser.add_argument("-n", "--notify", dest="Notify", default=False, action='store_true',
                            help='Sending notifications via a bot to a telegram chat')
    arg_parser.add_argument("-i", "--individual", dest="Individual", default=False, action="store_true",
                            help="Flag to set if align-table should be renewed after each agent training.")

    arg_parser.add_argument("-c", "--config", dest="Config", type=str, nargs='+', default=None,
                            help='File containing configurations for the program following the specifications of the '
                                 'README.md')
    arg_parser.add_argument("-m", "--multi", dest="Multi", default=1, nargs=1, type=int,
                            help='Set to use multithreading. Only available together with -c/--config or multiple '
                                 'benchmarks. An argument of one will be handled as running on all available cores. '
                                 'In order to run only on one core, just don\'t provide this argument.')
    arg_parser.add_argument("-b", "--benchmarks", dest="Benchmarks", type=str, nargs='+', default=[],
                            help='Benchmarks to use, given either as files or as the internal benchmark-annotation')
    arg_parser.add_argument("-f", "--folder", dest="Folder", default=None, type=str, nargs=1,
                            help="folder to store resulting alignments in FASTA-format in")
    arg_parser.add_argument("-u", "--update", dest="Update", default=True, action='store_false',
                            help="Flag to disable storing the best configuration per benchmark in a json file")
    arg_parser.add_argument("-e", "--seed", dest="Seed", default=None, type=int,
                            help="Seed for PyTorch to get reproducible results")
    arg_parser.add_argument("-M", "--similarity-matrix", dest="Similarity", default=False, action='store_true',
                            help="Flag to be set if one wants to compute the similarity matrix of the benchmark-files")
    return arg_parser


if __name__ == "__main__":
    warnings.simplefilter("error", np.VisibleDeprecationWarning)

    start = time.time()
    # create the argument parser and parse the provided arguments
    parser = get_arg_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        exit(0)

    print("reading parameters")
    result = parser.parse_args(sys.argv[1:])

    if result.Folder is not None:
        result.Folder = result.Folder[0]
    if isinstance(result.Multi, list):
        result.Multi = result.Multi[0]
    if result.Optimize == "SP":
        result.Optimize = SP_SCORE
    else:
        result.Optimize = C_SCORE
    if result.Seed is not None:
        torch.manual_seed(result.Seed)

    print("parse configurations")
    # process the arguments into
    if result.Config is not None:
        configurations, benchmarks = join_config_files([parse_config_file(config) for config in result.Config])
    else:
        configurations = [parse_arguments(result)]
        benchmarks = {(benchs[b] if b in benchs else b): list(range(len(configurations))) for b in result.Benchmarks}

    if result.Similarity:
        print("compute identity matrices")
        similarity_matrix(benchmarks)
        exit(0)

    print("start training")
    # start the read part of the program
    if result.BruteForce is not None and len(result.Benchmarks) != 0:
        multiprocessing_brute_force(benchmarks, result)
    else:
        run_multiple_agents(benchmarks, configurations, result,
                            multithreading=(result.Multi != 1 and (result.Config is not None or len(benchmarks) > 1)))

    time = time.time() - start

    if result.Notify:
        notify("Running finished within " + str(time) + " seconds.")
    print("Used time:", time)
