from dataclasses import dataclass, field

from networks.actorcritic_networks import GenericActorCriticNetwork
from networks.alphazero_networks import GenericAlphaZeroNetwork, A0_Network
from networks.ffnn_networks import GenericFFNN, ValueFFNN
from networks.reinforce_networks import GenericREINFORCENetwork, REINFORCENetwork
from utils.constants import TABLE_AGENT, ACTOR_CRITIC_AGENT, MCTS_AGENT, POLICY_AGENT, VALUE_AGENT, ALPHA_ZERO_AGENT, \
    SP_SCORE, C_SCORE

default_parameter = {
    "T": {"Games": 0, "StepsEpsilon": 0, "EpsilonEnd": 0.1, "Alpha": 0.8, "Gamma": 0.95, "Lambda": 0.1, "N": 0,
          "Score": SP_SCORE, "Look": False, "Support": True, "Progress": True, "Graph": False, "Notify": False,
          "Refinement": False},
    "V": {"Games": 0, "UpdateSteps": 0, "StepsEpsilon": 0, "EpsilonEnd": 0.1, "BatchSize": 32, "Alpha": 0.8,
          "Gamma": 0.95, "Lambda": 0.1, "N": 0, "Score": SP_SCORE, "Look": False, "Support": True, "Progress": True,
          "Graph": False, "Notify": False, "Refinement": False},
    "P": {"Games": 0, "StepsEpsilon": 0, "EpsilonEnd": 0.1, "Alpha": [0.01, 0.01], "Gamma": [0.99, 0.99],
          "Score": SP_SCORE, "Baseline": False, "Look": False, "Support": True, "Progress": True, "Graph": False,
          "Notify": False, "Refinement": False},
    "A": {"Games": 0, "StepsEpsilon": 0, "EpsilonEnd": 0.1, "Alpha": [0.01, 0.01], "Gamma": [0.99, 0.99],
          "Score": SP_SCORE, "Look": False, "Support": True, "Progress": True, "Graph": False, "Notify": False,
          "Refinement": False},
    "M": {"Simulations": 0, "Rollouts": 1, "C": 1.0, "Score": SP_SCORE, "Progress": False, "Notify": False,
          "Refinement": False, "Adjust": True},
    "0": {"Games": 0, "Simulations": 0, "Rollouts": 1, "C": 1, "Score": SP_SCORE, "Progress": False, "Graph": False,
          "Notify": False, "Refinement": False, "Adjust": True}
}
dp = default_parameter


def get_score(score):
    if isinstance(score, str) and score[0] == "C":
        return C_SCORE
    return SP_SCORE


@dataclass
class TableAgentTrainingConfiguration:
    """
    Wrapper class to keep the learning parameters for an instance of the TableAgent
    """
    games: int
    steps_epsilon: int
    epsilon_end: int
    alpha: float
    gamma: float
    lamb: float
    n: int
    score: int
    look_ahead_search: bool
    supported_search: bool

    print_console: bool
    print_graph: bool
    notification: bool
    refinement: bool

    name: str
    id: int = field(init=False, default=TABLE_AGENT)

    def __dict__(self):
        return {
            "Name": "Table",
            "Games": self.games,
            "StepsEpsilon": self.steps_epsilon,
            "EpsilonEnd": self.epsilon_end,
            "Alpha": self.alpha,
            "Gamma": self.gamma,
            "Lambda": self.lamb,
            "N": self.n,
            "Score": "SP" if self.score == SP_SCORE else "C",
            "Look": self.look_ahead_search,
            "Support": self.supported_search
        }


def table_config_from_dict(agent):
    """
    Transform a dictionary containing some (not necessarily all parameter)
    into a training configuration of a table agent
    :param agent: dictionary to parameterize the agent training
    :return: table_agent training_configuration
    """
    return TableAgentTrainingConfiguration(
        agent.get("Games", dp["T"]["Games"]), agent.get("StepsEpsilon", dp["T"]["StepsEpsilon"]),
        agent.get("EpsilonEnd", dp["T"]["EpsilonEnd"]), agent.get("Alpha", dp["T"]["Alpha"]),
        agent.get("Gamma", dp["T"]["Gamma"]), agent.get("Lambda", dp["T"]["Lambda"]), agent.get("N", dp["T"]["N"]),
        get_score(agent.get("Score", dp["T"]["Score"])), agent.get("Look", dp["T"]["Look"]),
        agent.get("Support", dp["T"]["Support"]), agent.get("Progress", dp["T"]["Progress"]),
        agent.get("Graph", dp["T"]["Graph"]), agent.get("Notify", dp["T"]["Notify"]),
        agent.get("Refinement", dp["T"]["Refinement"]), agent["Name"])


@dataclass
class ValueAgentTrainingConfiguration:
    """
    Wrapper class to keep the learning parameters for an instance of the ValueAgent
    """
    games: int
    update_steps: int
    steps_epsilon: int
    epsilon_end: int
    batch_size: int
    alpha: float
    gamma: float
    lamb: float
    n: int
    score: int
    network: GenericFFNN
    look_ahead_search: bool
    supported_search: bool

    print_console: bool
    print_graph: bool
    notification: bool
    refinement: bool

    name: str
    id: int = field(init=False, default=VALUE_AGENT)

    def __dict__(self):
        return {
            "Name": "Value",
            "Games": self.games,
            "UpdateSteps": self.update_steps,
            "StepsEpsilon": self.steps_epsilon,
            "EpsilonEnd": self.epsilon_end,
            "BatchSize": self.batch_size,
            "Alpha": self.alpha,
            "Gamma": self.gamma,
            "Lambda": self.lamb,
            "N": self.n,
            "Score": "SP" if self.score == SP_SCORE else "C",
            "Look": self.look_ahead_search,
            "Support": self.supported_search
        }


def value_config_from_dict(agent):
    """
    Transform a dictionary containing some (not necessarily all) parameter
    into a training configuration of a value agent
    :param agent: dictionary to parameterize the agent training
    :return: value_agent training_configuration
    """
    return ValueAgentTrainingConfiguration(
        agent.get("Games", dp["V"]["Games"]), agent.get("UpdateSteps", dp["V"]["UpdateSteps"]),
        agent.get("StepsEpsilon", dp["V"]["StepsEpsilon"]), agent.get("EpsilonEnd", dp["V"]["EpsilonEnd"]),
        agent.get("BatchSize", dp["V"]["BatchSize"]), agent.get("Alpha", dp["V"]["Alpha"]),
        agent.get("Gamma", dp["V"]["Gamma"]), agent.get("Lambda", dp["V"]["Lambda"]), agent.get("N", dp["V"]["N"]),
        get_score(agent.get("Score", dp["V"]["Score"])), ValueFFNN, agent.get("Look", dp["V"]["Look"]),
        agent.get("Support", dp["V"]["Support"]), agent.get("Progress", dp["V"]["Progress"]),
        agent.get("Graph", dp["V"]["Graph"]), agent.get("Notify", dp["V"]["Notify"]),
        agent.get("Refinement", dp["V"]["Refinement"]), agent["Name"])


@dataclass
class PolicyAgentTrainingConfiguration:
    """
    Wrapper class to keep the learning parameters for an instance of the PolicyAgent
    """
    games: int
    steps_epsilon: int
    epsilon_end: int
    value_alpha: float
    value_gamma: float
    policy_alpha: float
    policy_gamma: float
    baseline: bool
    score: int
    network: GenericREINFORCENetwork
    look_ahead_search: bool
    supported_search: bool

    print_console: bool
    print_graph: bool
    notification: bool
    refinement: bool

    name: str
    id: int = field(init=False, default=POLICY_AGENT)

    def __dict__(self):
        return {
            "Name": "Policy",
            "Games": self.games,
            "StepsEpsilon": self.steps_epsilon,
            "EpsilonEnd": self.epsilon_end,
            "Alpha": [self.value_alpha, self.policy_alpha],
            "Gamma": [self.value_gamma, self.policy_gamma],
            "Baseline": self.baseline,
            "Score": "SP" if self.score == SP_SCORE else "C",
            "Look": self.look_ahead_search,
            "Support": self.supported_search
        }


def policy_config_from_dict(agent):
    """
    Transform a dictionary containing some (not necessarily all) parameter
    into a training configuration of a policy agent
    :param agent: dictionary to parameterize the agent training
    :return: policy_agent training_configuration
    """
    return PolicyAgentTrainingConfiguration(
        agent.get("Games", dp["P"]["Games"]), agent.get("StepsEpsilon", dp["P"]["StepsEpsilon"]),
        agent.get("EpsilonEnd", dp["P"]["EpsilonEnd"]), agent.get("Alpha", dp["P"]["Alpha"])[0],
        agent.get("Gamma", dp["P"]["Gamma"])[0], agent.get("Alpha", dp["P"]["Alpha"])[1],
        agent.get("Gamma", dp["P"]["Gamma"])[1], agent.get("Baseline", dp["P"]["Baseline"]),
        get_score(agent.get("Score", dp["P"]["Score"])), REINFORCENetwork, agent.get("Look", dp["P"]["Look"]),
        agent.get("Support", dp["P"]["Support"]), agent.get("Progress", dp["P"]["Progress"]),
        agent.get("Graph", dp["P"]["Graph"]), agent.get("Notify", dp["P"]["Notify"]),
        agent.get("Refinement", dp["P"]["Refinement"]), agent["Name"])


@dataclass
class ActorCriticAgentTrainingConfiguration:
    """
    Wrapper class to keep the learning parameters for an instance of the ActorCriticAgent
    """
    games: int
    steps_epsilon: int
    epsilon_end: int
    value_alpha: float
    value_gamma: float
    policy_alpha: float
    policy_gamma: float
    score: int
    network: GenericActorCriticNetwork
    look_ahead_search: bool
    supported_search: bool

    print_console: bool
    print_graph: bool
    notification: bool
    refinement: bool

    name: str
    id: int = field(init=False, default=ACTOR_CRITIC_AGENT)

    def __dict__(self):
        return {
            "Name": "ActorCritic",
            "Games": self.games,
            "StepsEpsilon": self.steps_epsilon,
            "EpsilonEnd": self.epsilon_end,
            "Alpha": [self.value_alpha, self.policy_alpha],
            "Gamma": [self.value_gamma, self.policy_gamma],
            "Score": "SP" if self.score == SP_SCORE else "C",
            "Look": self.look_ahead_search,
            "Support": self.supported_search
        }


def actor_critic_config_from_dict(agent):
    """
    Transform a dictionary containing some (not necessarily all) parameter
    into a training configuration of a actor_critic agent
    :param agent: dictionary to parameterize the agent training
    :return: actor_critic_agent training_configuration
    """
    return ActorCriticAgentTrainingConfiguration(
        agent.get("Games", dp["A"]["Games"]), agent.get("StepsEpsilon", dp["A"]["StepsEpsilon"]),
        agent.get("EpsilonEnd", dp["A"]["EpsilonEnd"]), agent.get("Alpha", dp["A"]["Alpha"])[0],
        agent.get("Gamma", dp["A"]["Gamma"])[0], agent.get("Alpha", dp["A"]["Alpha"])[1],
        agent.get("Gamma", dp["A"]["Gamma"])[1], get_score(agent.get("Score", dp["A"]["Score"])), REINFORCENetwork,
        agent.get("Look", dp["A"]["Look"]), agent.get("Support", dp["A"]["Support"]),
        agent.get("Progress", dp["A"]["Progress"]), agent.get("Graph", dp["A"]["Graph"]),
        agent.get("Notify", dp["A"]["Notify"]), agent.get("Refinement", dp["A"]["Refinement"]), agent["Name"])


@dataclass
class MCTSTrainingConfiguration:
    """
    Wrapper class to keep the learning parameters for an instance of the MCTS-Agent
    """
    simulations: int
    rollouts: int
    c: float
    score: int

    print_console: bool
    notification: bool
    refinement: bool
    adjust: bool

    name: str
    id: int = field(init=False, default=MCTS_AGENT)

    def __dict__(self):
        return {
            "Name": "MCTS",
            "Simulations": self.simulations,
            "Rollouts": self.rollouts,
            "C": self.c,
            "Score": "SP" if self.score == SP_SCORE else "C",
            "Refinement": self.refinement,
            "Adjust": self.adjust
        }


def mcts_config_from_dict(agent):
    """
    Transform a dictionary containing some (not necessarily all) parameter
    into a training configuration of a mcts agent
    :param agent: dictionary to parameterize the agent training
    :return: mcts_agent training_configuration
    """
    return MCTSTrainingConfiguration(
        agent.get("Simulations", dp["M"]["Simulations"]), agent.get("Rollouts", dp["M"]["Rollouts"]),
        agent.get("C", dp["M"]["C"]), get_score(agent.get("Score", dp["M"]["Score"])),
        agent.get("Progress", dp["M"]["Progress"]), agent.get("Notify", dp["M"]["Notify"]),
        agent.get("Refinement", dp["M"]["Refinement"]), agent.get("Adjust", dp["M"]["Adjust"]), agent["Name"])


@dataclass
class AlphaZeroTrainingConfiguration:
    """
    Wrapper class to keep the learning parameters for an instance of the AlphaZeroAgent
    """
    games: int
    simulations: int
    rollouts: int
    c: float
    score: int
    network: GenericAlphaZeroNetwork

    print_console: bool
    print_graph: bool
    notification: bool
    refinement: bool
    adjust: bool

    name: str
    id: int = field(init=False, default=ALPHA_ZERO_AGENT)

    def __dict__(self):
        return {
            "Name": "Alpha",
            "Games": self.games,
            "Simulations": self.simulations,
            "Rollouts": self.rollouts,
            "C": self.c,
            "Score": "SP" if self.score == SP_SCORE else "C",
            "Refinement": self.refinement,
            "Adjust": self.adjust
        }


def alpha_zero_from_dict(agent):
    """
    Transform a dictionary containing some (not necessarily all) parameter
    into a training configuration of a alph-zero agent
    :param agent: dictionary to parameterize the agent training
    :return: alpha-zero agent training_configuration
    """
    return AlphaZeroTrainingConfiguration(
        agent.get("Games", dp["0"]["Games"]), agent.get("Simulations", dp["0"]["Simulations"]),
        agent.get("Rollouts", dp["0"]["Rollouts"]), agent.get("C", dp["0"]["C"]),
        get_score(agent.get("Score", dp["0"]["Score"])), A0_Network, agent.get("Progress", dp["0"]["Progress"]),
        agent.get("Graph", dp["0"]["Graph"]), agent.get("Notify", dp["0"]["Notify"]),
        agent.get("Refinement", dp["0"]["Refinement"]), agent.get("Adjust", dp["0"]["Adjust"]), agent["Name"])


def from_dict(agent):
    """
    Wrapper class to handle all sorts of parametrizations in dictionaries at once
    :param agent: dictionary containing the agents parameters
    :return: configuration according to input dictionary
    """
    if agent["Name"] == "Table":
        return table_config_from_dict(agent)
    elif agent["Name"] == "Value":
        return value_config_from_dict(agent)
    elif agent["Name"] == "Policy":
        return policy_config_from_dict(agent)
    elif agent["Name"] == "ActorCritic":
        return actor_critic_config_from_dict(agent)
    elif agent["Name"] == "MCTS":
        return mcts_config_from_dict(agent)
    elif agent["Name"] == "Alpha":
        return alpha_zero_from_dict(agent)
    else:
        return None
