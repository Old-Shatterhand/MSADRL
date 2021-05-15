import numpy as np
import torch
from torch import optim

from agent_training.PolicyTrainer import PolicyTrainer
from agents.actor_critic_agent import ActorCriticAgent
from networks.actorcritic_networks import TinyACNetwork
from utils.constants import SP_SCORE
from utils.profile import Profile
from utils.utils import get_sequences


class ActorCriticAgentTrainer(PolicyTrainer):
    def __init__(self, training_agent: ActorCriticAgent, games=0, steps_epsilon=0, epsilon_end=0.1, value_alpha=0.001,
                 policy_alpha=0.001, value_gamma=0.99, policy_gamma=0.99, score=SP_SCORE, look_ahead_search=False,
                 supported_search=False, refinement=False, data_name="SetX"):
        """
        Initializing the trainer with the nets and the additional fields used to train
        :param training_agent: initialized object of the agent that should be trained
        :param games: games to play while training
        :param steps_epsilon: count how many steps to perform until epsilon shall reach epsilon end
        :param epsilon_end: final value of epsilon at the end og epsilon decreasing
        :param value_alpha: learning_rate used to weight the loss when performing the weights update for value-approx.
        :param policy_alpha: learning_rate used to weight the loss when performing the weights update for policy-approx.
        :param value_gamma: discount factor to control influence of future reward on weights update for value-approx.
        :param policy_gamma: discount factor to control influence of future reward on weights update for policy-approx.
        :param score: score to optimize the trained agent for
        :param look_ahead_search: bool flag to control usage of look-ahead-search to guide the training process
        :param supported_search: control flag to enable search support by an array containing all possible next states
                                 disables reselection of an previously selected action
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param data_name: Name of dataset the agent in trained on
        """
        super(ActorCriticAgentTrainer, self).__init__(
            training_agent=training_agent, games=games, steps_epsilon=steps_epsilon, epsilon_end=epsilon_end,
            value_alpha=value_alpha, policy_alpha=policy_alpha, value_gamma=value_gamma, policy_gamma=policy_gamma,
            score=score, look_ahead_search=look_ahead_search, supported_search=supported_search, refinement=refinement,
            data_name=data_name)

        self.optimizer = optim.Adam(self.net.parameters())

    def learn(self, replay, steps=0):
        """
        Learns from the provided replay aka information about the previously played game.
        :param replay: data from the played game before this learning step
            consists of (states, actions, rewards, new_states, done, values and log_probs)
        :param steps: not used in this implementation
        :returns: the loss computed according to the A2C algorithm
        """
        if len(replay) == 0:
            return

        states, actions, rewards, new_states, dones, values, log_probs = zip(*replay)

        # compute the q_values based on the reward and the state_value of the last state
        q_val, _ = self.net.forward(new_states[-1])
        q_val = q_val.item()
        q_vals = np.zeros(len(values))
        for t in reversed(range(len(rewards))):
            q_val = rewards[t] + self.value_gamma * q_val
            q_vals[t] = q_val
        q_vals = torch.FloatTensor(q_vals)

        values = torch.FloatTensor(values)
        log_probs = torch.stack(log_probs)

        # compute the loss according to the A2C-Algorithm
        advantage = q_vals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        # adapt the weights according to the computed loss
        self.optimizer.zero_grad()
        ac_loss.backward()
        self.optimizer.step()

        # return current loss, used for hyperparameter optimization and checking of the training process
        return actor_loss.item(), critic_loss.item()


if __name__ == "__main__":  # this file was run from the command line
    print("#############################################")
    print("##Starting training of a actor-critic agent##")
    print("#############################################")
    print()

    score = SP_SCORE

    agent = ActorCriticAgent(sequences=get_sequences(count=3, length=6, different=True), network_object=TinyACNetwork)
    acat = ActorCriticAgentTrainer(training_agent=agent, supported_search=True)
    acat.run()

    # compute the resulting multiple sequence alignment
    (best_profile, best_permutation), _ = acat.evaluate_training()
    reward = best_profile.score()
    print(str(best_profile))
    print("Score:", reward[score], F"({best_permutation})")

    start = Profile(["ctattg", "ctaccg", "ctatgt"])
    print(start)
    print("Score:", start.score()[score])
    agent = ActorCriticAgent(sequences=start, network_object=TinyACNetwork, refinement=True)
    acat = ActorCriticAgentTrainer(agent, refinement=True)
    acat.run(True, False)
    (best_ref_profile, best_ref_permutation), _ = acat.evaluate_training()
    print(str(best_ref_profile))
    reward = best_ref_profile.score()
    print("Score:", reward[score], F"({best_ref_permutation})")
