import os
import time
from datetime import datetime
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import List

from agent_training.train_actor_critic_agent import ActorCriticAgentTrainer
from agent_training.train_alpha_zero_agent import AlphaZeroAgentTrainer
from agent_training.train_policy_agent import PolicyAgentTrainer
from agent_training.train_table_agent import TableAgentTrainer
from agent_training.train_value_agent import ValueAgentTrainer
from agents.actor_critic_agent import ActorCriticAgent
from agents.alpha_zero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from agents.policy_agent import PolicyAgent
from agents.table_solver import TableSolver
from agents.value_agent import ValueAgent
from utils.alignment import align_progressive, align_iterative
from utils.configurations import TableAgentTrainingConfiguration as TATC, ValueAgentTrainingConfiguration as VATC, \
    PolicyAgentTrainingConfiguration as PATC, ActorCriticAgentTrainingConfiguration as ACATC, \
    MCTSTrainingConfiguration as MCTSTC, AlphaZeroTrainingConfiguration as A0TC, from_dict
from utils.constants import names, benchmarks, seq_files, types, sizes, RL, DRL, MAFFT, CLUSTALW, MUSCLE, \
    VALUE_AGENT, TABLE_AGENT, POLICY_AGENT, MCTS_AGENT, ACTOR_CRITIC_AGENT, C_SCORE, SP_SCORE
from utils.hash_align_table import HashAlignTable
from utils.profile import Profile
from utils.utils import score_learning, read_fasta_data, get_sequence_type, get_sequence_size, output_learning, \
    notify, compare_alignments, get_score, get_from_config, get_c_type, write_best, read_best_file
from utils.wrapper import AlignmentWrapper, RefinementWrapper

number = 1


def optimize_agent(seqs, configurations, print_console, print_graph, create_agent_trainer):
    """
    Optimize one agent in different parametrization on the provided sequences from a benchmark
    :param seqs: sequences to optimize the agents on
    :param configurations: parameter configurations to be tested for optimality
    :param print_console: print log of training to the console
    :param print_graph: output the performance of each configurations as a graph
    :param create_agent_trainer: method to create the agent from the configurations and the according trainer
    :return: align-table with new alignments, optimal configuration and the according results (score, profile) and
        the scores of all other non-optimal configurations in the order they where tested
    """
    align_tables = {"SP": {"Refinement": HashAlignTable(Profile(seqs)), "Progressive": HashAlignTable(seqs)},
                    "CS": {"Refinement": HashAlignTable(Profile(seqs)), "Progressive": HashAlignTable(seqs)}}
    best_config = None
    best_score = (0, (0, 0))
    best_profile = None
    profile_scores = []

    # try all given configurations to fine the best performing one
    for config in configurations:
        agent, agent_trainer = create_agent_trainer(seqs, config)

        # train the agent ...
        agent_trainer.set_align_table(align_tables[get_score(config)][get_c_type(config)])
        performance = agent_trainer.run(print_console, print_graph)
        align_tables[get_score(config)][get_c_type(config)] = agent_trainer.get_align_table()

        # ... and compute the final alignment
        (profile, permutation), _ = agent_trainer.evaluate_training()
        profile_scores.append(profile.score())

        # check for being better than the actual best configuration
        act_score = (profile.score()[config.score], score_learning(*performance))
        if act_score > best_score:
            best_score = act_score
            best_config = config
            best_profile = profile
    return align_tables, best_config, best_score, best_profile, profile_scores


def create_table_agent(seqs, config: TATC):
    """
    creates a table agents and the according table-agent trainer
    :param seqs: sequences to be trained on
    :param config: configuration to use to instantiate the agent and the trainer
    :return: agent and trainer with the configurations
    """
    agent = TableSolver(seqs, config.refinement)
    agent_trainer = TableAgentTrainer(
        agent, games=config.games, steps_epsilon=config.steps_epsilon, epsilon_end=config.epsilon_end,
        alpha=config.alpha, gamma=config.gamma, lamb=config.lamb, n=config.n, score=config.score,
        look_ahead_search=config.look_ahead_search, supported_search=config.supported_search,
        refinement=config.refinement)
    return agent, agent_trainer


def create_value_agent(seqs, config: VATC):
    """
    creates a value agents and the according table-agent trainer
    :param seqs: sequences to be trained on
    :param config: configuration to use to instantiate the agent and the trainer
    :return: agent and trainer with the configurations
    """
    agent = ValueAgent(seqs, network_object=config.network, refinement=config.refinement)
    agent_trainer = ValueAgentTrainer(
        agent, games=config.games, update_steps=config.update_steps, steps_epsilon=config.steps_epsilon,
        epsilon_end=config.epsilon_end, batch_size=config.batch_size, alpha=config.alpha, gamma=config.gamma,
        lamb=config.lamb, n=config.n, score=config.score, look_ahead_search=config.look_ahead_search,
        supported_search=config.supported_search, refinement=config.refinement)
    return agent, agent_trainer


def create_policy_agent(seqs, config: PATC):
    """
    creates a policy agents and the according table-agent trainer
    :param seqs: sequences to be trained on
    :param config: configuration to use to instantiate the agent and the trainer
    :return: agent and trainer with the configurations
    """
    agent = PolicyAgent(seqs, config.network, config.refinement)
    agent_trainer = PolicyAgentTrainer(
        agent, games=config.games, steps_epsilon=config.steps_epsilon, epsilon_end=config.epsilon_end,
        value_alpha=config.value_alpha, policy_alpha=config.policy_alpha, value_gamma=config.value_gamma,
        policy_gamma=config.policy_gamma, baseline=config.baseline, score=config.score,
        look_ahead_search=config.look_ahead_search, supported_search=config.supported_search,
        refinement=config.refinement)
    return agent, agent_trainer


def create_actor_critic_agent(seqs, config: ACATC):
    """
    creates an actor critic agents and the according table-agent trainer
    :param seqs: sequences to be trained on
    :param config: configuration to use to instantiate the agent and the trainer
    :return: agent and trainer with the configurations
    """
    agent = ActorCriticAgent(seqs, config.network, refinement=config.refinement)
    agent_trainer = ActorCriticAgentTrainer(
        agent, games=config.games, steps_epsilon=config.steps_epsilon, epsilon_end=config.epsilon_end,
        value_alpha=config.value_alpha, policy_alpha=config.policy_alpha, value_gamma=config.value_gamma,
        policy_gamma=config.policy_gamma, score=config.score, look_ahead_search=config.look_ahead_search,
        supported_search=config.supported_search, refinement=config.refinement)
    return agent, agent_trainer


def optimize_mcts_agent(seqs, configurations: List[MCTSTC], align_table=None):
    """
    Optimize the MCTS learning agent over the given configurations of hyperparameters
    :param seqs: sequences to use for alignment
    :param configurations: configurations to test while training
    :param align_table: table containing previously computed alignments
    :return: table containing the computed alignments and the results of the aligning
    """
    align_tables = {"SP": {"Refinement": HashAlignTable(Profile(seqs)), "Progressive": HashAlignTable(seqs)},
                    "CS": {"Refinement": HashAlignTable(Profile(seqs)), "Progressive": HashAlignTable(seqs)}}
    best_config = None
    best_score = 0

    # try all defined configurations of hyperparameter to find the best performing one
    for config in configurations:
        agent = MCTSAgent(seqs, simulations=config.simulations, rollouts=config.rollouts, c=config.c,
                          score=config.score, refinement=config.refinement, adjust=config.adjust)
        agent.set_align_table(align_tables[get_score(config)][get_c_type(config)])
        if config.refinement:
            pass
        else:
            # compute the alignment using UCT-MCTS to find the most promising sequence
            env = AlignmentWrapper(seqs, agent, config.score)
            score, _, _, done = env.run()
            align_tables[get_score(config)][get_c_type(config)] = agent.get_align_table()

            if score > best_score and done:
                best_score = score
                best_config = config
    return align_table, best_config, best_score


def create_alpha_zero_agent(seqs, config: A0TC):
    """
    creates an actor critic agents and the according table-agent trainer
    :param seqs: sequences to be trained on
    :param config: configuration to use to instantiate the agent and the trainer
    :return: agent and trainer with the configurations
    """
    agent = AlphaZeroAgent(seqs, config.network, refinement=config.refinement)
    agent_trainer = AlphaZeroAgentTrainer(
        agent, games=config.games, simulations=config.simulations, rollouts=config.rollouts, c=config.c,
        score=config.score, refinement=config.refinement, adjust=config.adjust)
    return agent, agent_trainer


def run_agent(config, sequences, align_table, name, best_score, updating, total, individual, data_file):
    """
    run the by config specified agent on the sequences and use the given align-table as util to speed up th alignment
    :param config: configuration of the agent to train
    :param sequences: sequences of the benchmark to run the agent on
    :param align_table: table to use for alignment-speedup
    :param name: name of the benchmark, needed for the reports after training
    :param best_score: best_score ever reached on this benchmark
    :param updating: flag indicating to update the json file
    :param total: total number of trainings performed in this run
    :param individual: flag to indicate the use of a fresh align-table
    :param data_file: file to store all data about the computations
    :return: - the message to print and to sent via Telegram,
             - the score reached by the agent,
             - the extended align_table with probably new alignments,
             - the profile computed by this algorithm that leads to the result score and
             - the permutation of the input sequences to get the resulting profile
    """
    print(F"train {config.name} agent result on {name}")

    global number

    if config.id == MCTS_AGENT:
        agent = MCTSAgent(sequences, simulations=config.simulations, rollouts=config.rollouts, c=config.c,
                          score=config.score, refinement=config.refinement, console=config.print_console,
                          adjust=config.adjust)
        if not individual:
            agent.set_align_table(align_table)

        # compute the alignment using UCT-MCTS to find the most promising sequence
        env = RefinementWrapper(sequences, agent, config.score) if config.refinement else AlignmentWrapper(
            sequences, agent)
        try:
            start = time.time()
            _, permutation, profile, _ = env.run()
            end = time.time()
            runtime = end - start
        except:
            print("MCTS-Agent crashed! Return standard results.")
            permutation = [] if config.refinement else list(range(len(sequences)))
            profile = sequences if config.refinement else align_progressive(permutation, sequences)
            runtime = 0

        align_table = agent.get_align_table()
        print("Align-Table-Stats:", align_table.stats)

        # extract the score
        score = profile.score() if profile is not None else (0, 0)
        scoring = (score[0], score[1], "-", "-")

        # create the results-message
        mode = "Refinement" if config.refinement else "Progressive"
        optimizing = "C" if config.score == C_SCORE else "SP"
        message = F"MCTS algorithm result on {name}:\n" \
                  F"\tMode: {mode}, Optimizing: {optimizing}-Score\n" \
                  F"\tProgress: {number}/{total} ({round((number / total) * 100, 2)}%)\n" \
                  F"\tRuntime: {runtime} seconds\n" \
                  F"\tbest Alignment found: {score[0]}, {round(score[1], 2)}\n"
        data_file.write(F"MCTS\t{name}\t{mode}\t{optimizing}\t{runtime}\t{score[0]}\t{round(score[1], 2)}\t[" +
                        ";".join([str(p) for p in permutation]) + "]\t[" +
                        ";".join([str(s) for s in align_table.stats]) + "]\n")

    else:
        # if not MCTS agent, create one of the other agents
        if config.id == TABLE_AGENT:
            agent, agent_trainer = create_table_agent(sequences, config)
        elif config.id == VALUE_AGENT:
            agent, agent_trainer = create_value_agent(sequences, config)
        elif config.id == POLICY_AGENT:
            agent, agent_trainer = create_policy_agent(sequences, config)
        elif config.id == ACTOR_CRITIC_AGENT:
            agent, agent_trainer = create_actor_critic_agent(sequences, config)
        else:
            agent, agent_trainer = create_alpha_zero_agent(sequences, config)

        # train the agent ...
        if not individual:
            agent_trainer.set_align_table(align_table)
        try:
            _, _, _, runtime = agent_trainer.run(config.print_console, config.print_graph)
            align_table = agent_trainer.get_align_table()
            print("Align-Table-Stats:", align_table.stats)

            # ... and compute the final alignment
            (profile, permutation), (reward, _, _, _) = agent_trainer.evaluate_training()
        except:
            print("Error occurred in training of ")
            permutation = list(range(len(sequences)))
            profile = align_progressive(permutation, sequences, align_table)
            reward = 0, 0
            runtime = 0

        # extract the score
        score = profile.score() if profile is not None else (0, 0)
        scoring = (score[0], score[1], reward[0], reward[1])

        # create the results-message
        mode = "Refinement" if config.refinement else "Progressive"
        optimizing = "C" if config.score == C_SCORE else "SP"
        message = F"{agent.name()} algorithm result on {name}:\n" \
                  F"\tMode: {mode}, Optimizing: {optimizing}-Score\n" \
                  F"\tProgress: {number}/{total} ({round((number / total) * 100, 2)}%)\n" \
                  F"\tRuntime: {runtime} seconds\n" \
                  F"\tbest Alignment found: {score[0]}, {round(score[1], 2)}\n" \
                  F"\tlast Alignment found: {reward[0]}, {round(reward[1], 2)}"
        data_file.write(F"{agent.name}\t{name}\t{mode}\t{optimizing}\t{runtime}\t{score[0]}\t{round(score[1], 2)}\t[" +
                        ";".join([str(p) for p in permutation]) + "]\t[" +
                        ";".join([str(s) for s in align_table.stats]) + "]\n")

    # increase counter for configs that have been tested
    number += 1

    # check, whether older alignment has been improved
    if updating and (score[config.score], score[1 - config.score]) > \
            (best_score[config.score], best_score[1 - config.score]):
        message = "NEW BEST ALIGNMENT FOUND!\n" + message

    # print the message of the results and notify via telegram
    print(message)
    if config.notification:
        notify(message)

    return message, scoring, align_table, profile, permutation


def initialize_benchmark(b_id, best):
    """
    Initialize the execution of any agents on the benchmarks by creating necessary fields needed for search and training
    :param b_id: benchmark-id to train on
    :param best: best result from previous training runs on that benchmark
    :return: - the name of the benchmark used for evaluation tables at the end
             - the sequences of the benchmark as list of stings
             - the names according to the sequences in the same order ( = permutation [0,1,...,n-1,n])
             - basic data of the benchmark-sequences, namely type, count and average length
             - basic comparison of available results for this benchmark
             - best result from previous run on this benchmark
    """
    '''
    Things like comparison of different agents, best-marks on benchmarks and align-tables are stored in such nested
    HashMaps and are accessed and modified according to the actually used configuration and its scoring and aligning
    
    The benchmark_best contains the best results on a benchmark for each alignment setting. Such an refinement-tuple 
    consists of the Profile from the iterative aligning, the according iterative permutation, the permutation for the 
    starting alignment and the configuration that led to the optimal alignment
    '''
    comparison = {"SP": {"Refinement": {}, "Progressive": {}}, "CS": {"Refinement": {}, "Progressive": {}}}
    benchmark_best = {"SP": {"Refinement": (Profile([]), None, None, None), "Progressive": (Profile([]), None, None)},
                      "CS": {"Refinement": (Profile([]), None, None, None), "Progressive": (Profile([]), None, None)}}
    # if the benchmark is of this work and known because it is used while development, extract data from constants
    if isinstance(b_id, int):
        name, b, seqs_file = names[b_id], benchmarks[b_id], seq_files[b_id]
        sequences, sequence_names = read_fasta_data(seqs_file)

        # Insert the base-data for each setting, this is the left-hand side of the tables outputted at the end
        comparison["SP"]["Progressive"] = {RL: b[0:2], DRL: b[2:4], CLUSTALW: b[4:6], MAFFT: b[6:8], MUSCLE: b[8:10]}
        comparison["CS"]["Progressive"] = {RL: b[0:2], DRL: b[2:4], CLUSTALW: b[4:6], MAFFT: b[6:8], MUSCLE: b[8:10]}
        comparison["SP"]["Refinement"] = {RL: b[0:2], DRL: b[2:4], CLUSTALW: b[4:6], MAFFT: b[6:8], MUSCLE: b[8:10]}
        comparison["CS"]["Refinement"] = {RL: b[0:2], DRL: b[2:4], CLUSTALW: b[4:6], MAFFT: b[6:8], MUSCLE: b[8:10]}
        base_data = (b_id, types[b_id], sizes[b_id])
    # else read in the benchmark sequences and compute the base data
    else:
        name, (sequences, sequence_names) = os.path.basename(b_id).split(".")[0], read_fasta_data(b_id)

        # again the base data as left-hand side of the output tables, but here with zeros as they are not performed
        comparison["SP"]["Progressive"] = {RL: (0, 0), DRL: (0, 0), CLUSTALW: (0, 0), MAFFT: (0, 0), MUSCLE: (0, 0)}
        comparison["CS"]["Progressive"] = {RL: (0, 0), DRL: (0, 0), CLUSTALW: (0, 0), MAFFT: (0, 0), MUSCLE: (0, 0)}
        comparison["SP"]["Refinement"] = {RL: (0, 0), DRL: (0, 0), CLUSTALW: (0, 0), MAFFT: (0, 0), MUSCLE: (0, 0)}
        comparison["CS"]["Refinement"] = {RL: (0, 0), DRL: (0, 0), CLUSTALW: (0, 0), MAFFT: (0, 0), MUSCLE: (0, 0)}
        base_data = (b_id, get_sequence_type(sequences), get_sequence_size(sequences))

    '''
    Find and fill in the best alignments per optimization setting that can be found for the individual benchmark in the 
    store of best alignments. If it is not known, the (Profile([]), None, None) tuple remains
    '''
    if name in best["SP"]["Progressive"]:
        tmp = best["SP"]["Progressive"][name]
        benchmark_best["SP"]["Progressive"] = (align_progressive(tmp["Permutation"], sequences), tmp["Permutation"],
                                               from_dict(tmp["Configuration"]))
    if name in best["CS"]["Progressive"]:
        tmp = best["CS"]["Progressive"][name]
        benchmark_best["CS"]["Progressive"] = (align_progressive(tmp["Permutation"], sequences), tmp["Permutation"],
                                               from_dict(tmp["Configuration"]))
    if name in best["SP"]["Refinement"]:
        tmp = best["SP"]["Refinement"][name]
        benchmark_best["SP"]["Refinement"] = \
            (align_iterative(tmp["Permutation"], align_progressive(tmp["BasePermutation"], sequences)),
             tmp["Permutation"], tmp["BasePermutation"], from_dict(tmp["Configuration"]))
    if name in best["CS"]["Refinement"]:
        tmp = best["CS"]["Refinement"][name]
        benchmark_best["CS"]["Refinement"] = \
            (align_iterative(tmp["Permutation"], align_progressive(tmp["BasePermutation"], sequences)),
             tmp["Permutation"], tmp["BasePermutation"], from_dict(tmp["Configuration"]))

    '''
    insert the baseline (aka score of best progressive alignment) into the results-table to see from which value the 
    agent stared its alignment and to be able to argue on whether the alignment has been improved or not
    '''
    comparison["SP"]["Refinement"][TABLE_AGENT] = (*(benchmark_best["SP"]["Progressive"][0].score()[0:2]), 0, 0)
    comparison["CS"]["Refinement"][TABLE_AGENT] = (*(benchmark_best["CS"]["Progressive"][0].score()[0:2]), 0, 0)

    return name, sequences, sequence_names, base_data, comparison, benchmark_best


def multithread_agents_on_benchmark(name, sequences, sequence_names, configurations, agent_ids, comparison,
                                    benchmark_best, settings, data_file):
    """
    Execute multiple agent trainings on a benchmark using multithreading
    :param name: name of the benchmark
    :param sequences: sequences of the benchmark
    :param sequence_names: names of the sequence in the same order as the sequences
    :param configurations: configurations of agents to train on the benchmark
    :param agent_ids: dictionary of the ids of the agents to train per benchmark
    :param comparison: HashMap containing the results of reference tool on the benchmarks for each of the four
        optimization settings in a nested HashMap, this is enlarged by the results computed by the agents in here
        Its a 3-fold nested HashMap of of result tuples, the keys are the score, the alignment and the agent
    :param benchmark_best: best result on the benchmark for all four optimization settings in a nested HashMap
    :param settings: settings for this optimization task
    :param data_file: file to store all data about the computations
    :return: - enlarged comparison of agent results on this benchmark
             - new (or old) best result of and optimization setting on this benchmark
             - flag indicating if any new best alignment has been found
    """
    global number
    print("train multiple agents on one benchmark")

    # instantiate the tools for multithreading
    pool = ThreadPool(processes=cpu_count() if settings.Multi == 1 else settings.Multi)
    tasks = [None for _ in range(len(configurations))]

    # initialize one hashtable per optimization for faster computation

    align_tables = {"SP": {"Refinement": HashAlignTable(benchmark_best["SP"]["Progressive"][0]),
                           "Progressive": HashAlignTable(sequences)},
                    "CS": {"Refinement": HashAlignTable(benchmark_best["CS"]["Progressive"][0]),
                           "Progressive": HashAlignTable(sequences)}}
    changed = False

    for i, config in enumerate(configurations):
        # if the agent is not to be executed on this benchmark insert this into the comparison
        if i not in agent_ids:
            number += 1
            if config.refinement:
                comparison[get_score(config)]["Refinement"][i + TABLE_AGENT + 1] = ("-", "-", "-", "-")
            else:
                comparison[get_score(config)]["Progressive"][i + TABLE_AGENT] = ("-", "-", "-", "-")
            continue

        # if the agent perform refinement ...
        if config.refinement:
            # ... check if this is possible ...
            if benchmark_best[get_score(config)]["Progressive"][1] is None:
                print("WARNING: Cannot compute Refinement without base-profile of progressive alignment")
                continue
            # ... and run this agent
            tasks[i] = pool.apply_async(run_agent, (config, benchmark_best[get_score(config)]["Progressive"][0],
                                                    get_from_config(align_tables, config), name, benchmark_best[
                                                        get_score(config)]["Refinement"][0].score(),
                                                    settings.Update, len(configurations), settings.Individual,
                                                    data_file))
        else:
            # otherwise run a progressive alignment
            tasks[i] = pool.apply_async(run_agent, (config, sequences, get_from_config(align_tables, config), name,
                                                    benchmark_best[get_score(config)]["Progressive"][0].score(),
                                                    settings.Update, len(configurations), settings.Individual,
                                                    data_file))

    for i, config in enumerate(configurations):
        # if the agent has not been trained, skip this
        if tasks[i] is None:
            continue

        # if selected insert the results of the computation into the refinement results
        if config.refinement:
            # extract the results from the background process
            message, scoring, refine_align_table, profile, permutation = tasks[i].get()

            # update the comparison
            get_from_config(comparison, config)[i + TABLE_AGENT + 1] = scoring

            # and update the best-marks on the benchmark and store whether a better alignment has been found
            benchmark_best[get_score(config)]["Refinement"], change = compare_alignments(get_from_config(
                benchmark_best, config), (profile, permutation, benchmark_best[get_score(config)]["Progressive"][1],
                                          config), config.score)
        # otherwise into the results of progressive alignment
        else:
            # extract the result from the background process ...
            message, scoring, prog_align_table, profile, permutation = tasks[i].get()

            # ... update the comparison ...
            get_from_config(comparison, config)[i + TABLE_AGENT] = scoring

            # ... and update the best-marks on the benchmark and store if a now best alignment has been found
            benchmark_best[get_score(config)]["Progressive"], change = compare_alignments(benchmark_best[get_score(
                config)]["Progressive"], (profile, permutation, config), config.score)

        # if specified the output has to be saved in the provided folder
        if settings.Folder is not None:
            profile.store(settings.Folder, i, config, name, sequence_names, permutation)
        changed |= change

    return comparison, benchmark_best, changed


def multithread_agent_on_benchmarks(benchmark_ids, configurations, best, settings, data_file):
    """
    run a single agent configuration on all benchmarks in parallel
    :param benchmark_ids: benchmark ids
    :param configurations: configuration(s)
    :param best: best results in the actual optimization setting
    :param settings: settings of this search
    :param data_file: file to store all data about the computations
    :return: - a bool flag indicating that a new best alignments has been found
             - the actual comparison of agents on different benchmarks
             - the actual best in JSON format
    """
    global number
    print("train one agent on multiple benchmarks")

    # initialize the multithreading tools needed
    pool = ThreadPool(processes=cpu_count() if settings.Multi == 1 else settings.Multi)
    tasks = [None for _ in range(len(benchmark_ids))]

    # set empty variables to use for the statistics of the computation
    name, sequences, sequence_names, base_data = "", [], [], ()
    comparison = {"SP": {"Refinement": [], "Progressive": []}, "CS": {"Refinement": [], "Progressive": []}}

    '''
    lists to store the essential data from the benchmarks per benchmark as they are not created in different disjoint 
    for-loops and therefore has to be conserved, this is done for the comparison and the best-marks
    '''
    names = [None for _ in range(len(benchmark_ids))]
    sequence_names = [None for _ in range(len(benchmark_ids))]
    base_dataset = [None for _ in range(len(benchmark_ids))]
    bench_comp = [None for _ in range(len(benchmark_ids))]
    bench_best = [None for _ in range(len(benchmark_ids))]
    config = configurations[0]
    changed = False

    for i, (b_id, agent_ids) in enumerate(benchmark_ids.items()):
        # initialize the files based on the actual benchmark
        names[i], sequences, sequence_names[i], base_dataset[i], bench_comp[i], bench_best[i] = \
            initialize_benchmark(b_id, best)

        # if the configuration is not to be used on this benchmark, insert into the statistics and continue with next
        if 0 not in agent_ids:
            number += 1
            if config.refinement:
                get_from_config(bench_comp[i], config)[TABLE_AGENT + 1] = ("-", "-", "-", "-")
            else:
                get_from_config(bench_comp[i], config)[TABLE_AGENT] = ("-", "-", "-", "-")
            continue

        # if indicated perform iterative refinement
        if config.refinement:
            if bench_best[i]["SP" if config.score == SP_SCORE else "CS"]["Progressive"][1] is None:
                print("WARNING: Cannot compute Refinement without basic profile of progressive alignment")
                continue
            tasks[i] = pool.apply_async(run_agent, (config, bench_best[i][get_score(config)]["Refinement"][0],
                                                    HashAlignTable(Profile(sequences)), names[i],
                                                    bench_best[i][get_score(config)]["Refinement"][0].score(),
                                                    settings.Update, len(configurations), settings.Individual,
                                                    data_file))
        else:
            # else compute a progressive alignment
            tasks[i] = pool.apply_async(run_agent, (config, sequences, HashAlignTable(sequences), names[i],
                                                    bench_best[i][get_score(config)]["Progressive"][0].score(),
                                                    settings.Update, len(configurations), settings.Individual,
                                                    data_file))

    # iterate over the processes and collect the results
    for i in range(len(benchmark_ids.items())):
        if config.refinement:
            # update the statistics of refinement analysis
            message, scoring, _, profile, permutation = tasks[i].get()
            get_from_config(bench_comp[i], config)[TABLE_AGENT + 1] = scoring
            bench_best[i][get_score(config)]["Refinement"], change = \
                compare_alignments(bench_best[i][get_score(config)]["Refinement"],
                                   (profile, permutation, bench_best[i][get_score(config)]["Progressive"][1], config),
                                   config.score)
        else:
            # or the results of progressive alignments
            message, scoring, _, profile, permutation = tasks[i].get()
            get_from_config(bench_comp[i], config)[TABLE_AGENT] = scoring
            bench_best[i][get_score(config)]["Progressive"], change = \
                compare_alignments(bench_best[i][get_score(config)]["Progressive"], (profile, permutation, config),
                                   config.score)

        # update the global statistics
        comparison, best = update_comparison(names[i], base_dataset[i], best, comparison, bench_comp[i], bench_best[i])

        if settings.Folder is not None:
            profile.store(settings.Folder, i, config, names[i], sequence_names[i], permutation)
        changed |= change

    return changed, comparison, best


def sequential_agents_on_benchmark(name, sequences, sequence_names, configurations, agent_ids, tmp_comparison,
                                   benchmark_best, settings, data_file):
    """
    run agents on the benchmarks sequential (not in parallel)
    :param name: name of the benchmark to train on
    :param sequences: sequences of the benchmark
    :param sequence_names: names of the sequences in the same order as the sequences
    :param configurations: configurations to train
    :param agent_ids: dict mapping agent ids on the benchmarks
    :param tmp_comparison: array of temporary results on this benchmark for each optimization setting
    :param benchmark_best: best results on this benchmark for each optimization setting
    :param settings: settings of this search
    :param data_file: file to store all data about the computations
    :return: - comparison of the agents on this benchmark used for later statistics for each optimization setting
             - best result on this benchmark for each optimization setting
             - a bool flag indicating that a new best alignment has been found
    """
    global number
    print("train agent(s) on benchmark(s) sequentially")

    align_tables = {"SP": {"Refinement": HashAlignTable(benchmark_best["SP"]["Progressive"][0]),
                           "Progressive": HashAlignTable(sequences)},
                    "CS": {"Refinement": HashAlignTable(benchmark_best["CS"]["Progressive"][0]),
                           "Progressive": HashAlignTable(sequences)}}
    changed = False

    for i, config in enumerate(configurations):
        # if the agent is not to execute on this benchmark, update statistics and skip
        if i not in agent_ids:
            number += 1
            if config.refinement:
                tmp_comparison[get_score(config)]["Refinement"][i + TABLE_AGENT + 1] = ("-", "-", "-", "-")
            else:
                tmp_comparison[get_score(config)]["Progressive"][i + TABLE_AGENT] = ("-", "-", "-", "-")
            continue

        # if indicated perform iterative refinement
        if config.refinement:
            # if not possible because there is no progressive base alignment, skip
            if benchmark_best[get_score(config)]["Progressive"][1] is None:
                print("WARNING: Cannot compute Refinement without base-profile of progressive alignment")
                continue

            message, scoring, align_tables[get_score(config)][get_c_type(config)], profile, permutation = \
                run_agent(config, benchmark_best[get_score(config)]["Progressive"][0],
                          get_from_config(align_tables, config), name,
                          benchmark_best[get_score(config)]["Refinement"][0].score(), settings.Update,
                          len(configurations), settings.Individual, data_file)

            # and update the statistics
            tmp_comparison[get_score(config)]["Refinement"][i + TABLE_AGENT + 1] = scoring
            benchmark_best[get_score(config)]["Refinement"], change = \
                compare_alignments(benchmark_best[get_score(config)]["Refinement"],
                                   (profile, permutation, benchmark_best[get_score(config)]["Progressive"][1], config),
                                   config.score)
        else:
            # otherwise a progressive alignment
            message, scoring, prog_align_table, profile, permutation = \
                run_agent(config, sequences, get_from_config(align_tables, config), name,
                          benchmark_best[get_score(config)]["Progressive"][0].score(), settings.Update,
                          len(configurations), settings.Individual, data_file)

            # ad update the statistics
            tmp_comparison[get_score(config)]["Progressive"][i + TABLE_AGENT] = scoring
            benchmark_best[get_score(config)]["Progressive"], change = \
                compare_alignments(benchmark_best[get_score(config)]["Progressive"],
                                   (profile, permutation, config), config.score)

        # if indicated save the results to the target-folder
        if settings.Folder is not None:
            profile.store(settings.Folder, i, config, name, sequence_names, permutation)
        changed |= change

    return tmp_comparison, benchmark_best, changed


def update_comparison(name, base_data, best, comparison, tmp_comparison, benchmark_best):
    """
    Update the comparison of agents on benchmarks
    :param name: name of the benchmark
    :param base_data: base-data of this benchmark
    :param best: collection of best results of the actual optimization setting JSON FORMATTED
    :param comparison: comparison of agents on different benchmarks
    :param tmp_comparison: new results of agents on the benchmark
    :param benchmark_best: best result on this benchmark
    :return: - extended comparison of agents on all benchmarks
             - updated best value of agents on each benchmark in JSON format
    """
    for score in ["SP", "CS"]:
        if tmp_comparison[score]["Progressive"] != 0:

            # update the progressive comparison
            comparison[score]["Progressive"].append((*base_data, tmp_comparison[score]["Progressive"]))

            # update the all-time best result of progressive agents
            if benchmark_best[score]["Progressive"][1] is not None:
                best[score]["Progressive"][name] = {
                    "Score": str(benchmark_best[score]["Progressive"][0].score()),
                    "Permutation": [int(x) for x in benchmark_best[score]["Progressive"][1]],
                    "Configuration": benchmark_best[score]["Progressive"][2].__dict__()
                }

        if tmp_comparison[score]["Refinement"] != 0:

            # update the refinement comparison
            comparison[score]["Refinement"].append((*base_data, tmp_comparison[score]["Refinement"]))

            # update the all-time best result of refinement agents
            if benchmark_best[score]["Refinement"][1] is not None:
                best[score]["Refinement"][name] = {
                    "Score": str(benchmark_best[score]["Refinement"][0].score()),
                    "Permutation": [int(x) for x in benchmark_best[score]["Refinement"][1]],
                    "BasePermutation": [int(x) for x in benchmark_best[score]["Refinement"][2]],
                    "Configuration": benchmark_best[score]["Refinement"][3].__dict__()
                }

    return comparison, best


def run_multiple_agents(benchmark_ids: dict, configurations: list, settings, multithreading=False):
    """
    run multiple agents on multiple benchmarks for progressive alignment and iterative refinement at once
    :param benchmark_ids: benchmarks to run on
    :param configurations: configurations to test on given benchmarks
    :param settings: settings storing hyperparameter of the search
    :param multithreading: flag indicating to use multithreading for faster computation of the alignments
    """
    global number
    comparison = {"SP": {"Refinement": [], "Progressive": []}, "CS": {"Refinement": [], "Progressive": []}}

    # read in the best results from previous runs
    best, current = read_best_file()
    data_file = open("./" + datetime.now().strftime("msadrl_data_%Y%m%d%H%M%S") + ".tsv", "w")

    # if multithreading is enabled and there is only one agent-configuration to run, execute this in parallel
    if multithreading and len(configurations) == 1:
        # run the agent
        changed, comparison, best = multithread_agent_on_benchmarks(benchmark_ids, configurations, best, settings,
                                                                    data_file)

        # save the results  if at least one results has been improved
        if settings.Update and changed:
            write_best(best, current)
    else:
        for b_id, agent_ids in benchmark_ids.items():
            number = 0
            # initialize the computation on this benchmark by setting the search-necessary fields
            name, sequences, sequence_names, base_data, tmp_comparison, benchmark_best = \
                initialize_benchmark(b_id, best)

            # if multithreading is enabled and there are more than one agents to train, execute them in parallel
            if multithreading and len(configurations) > 1:
                # get the results from the parallel execution
                tmp_comparison, benchmark_best, changed = \
                    multithread_agents_on_benchmark(name, sequences, sequence_names, configurations, agent_ids,
                                                    tmp_comparison, benchmark_best, settings, data_file)
            else:
                # otherwise get the results from the sequential execution
                tmp_comparison, benchmark_best, changed = \
                    sequential_agents_on_benchmark(name, sequences, sequence_names, configurations, agent_ids,
                                                   tmp_comparison, benchmark_best, settings, data_file)

            # update the statistics on this benchmark
            comparison, best = update_comparison(name, base_data, best, comparison, tmp_comparison, benchmark_best)

            # write the temporary results of this benchmark to the file if at least one results has been improved
            if settings.Update and changed:
                write_best(best, current)

    # printout the table of the progressive results
    if any([len(b[3]) > 5 for b in comparison["SP"]["Progressive"]]):
        print("\nProgressive SP-Score-Alignments:")
        output_learning(comparison["SP"]["Progressive"], configurations, names)

    if any([len(b[3]) > 5 for b in comparison["CS"]["Progressive"]]):
        print("\nProgressive C-Score-Alignments:")
        output_learning(comparison["CS"]["Progressive"], configurations, names)

    # printout the table of the refinement results
    if any([len(b[3]) > 6 for b in comparison["SP"]["Refinement"]]):
        print("\nIterative SP-Score-Refinements:")
        output_learning(comparison["SP"]["Refinement"], configurations, names, True)

    if any([len(b[3]) > 6 for b in comparison["CS"]["Refinement"]]):
        print("\nIterative C-Score-Refinements:")
        output_learning(comparison["CS"]["Refinement"], configurations, names, True)
