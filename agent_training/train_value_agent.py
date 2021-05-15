import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent_training.FunctionTrainer import FunctionTrainer
from agents.value_agent import ValueAgent
from networks.ffnn_networks import TinyValueFFNN
from utils.constants import MC_LEARN, LAMBDA_LEARN, SP_SCORE
from utils.prioritized_experience_replay_buffer import PrioritizedExperienceReplayBuffer as PrioERB
from utils.profile import Profile
from utils.utils import lambda_rewards, get_sequences


class ValueAgentTrainer(FunctionTrainer):
    def __init__(self, training_agent: ValueAgent, games=0, update_steps=0, steps_epsilon=0, epsilon_end=0.0,
                 batch_size=32, alpha=0.001, gamma=0.9, n=MC_LEARN, lamb=0.1, score=SP_SCORE, look_ahead_search=False,
                 supported_search=False, refinement=False, data_name="SetX"):
        """
        Initializes an agent trainer for value agent-training and all needed fields.
        :param training_agent: the agent to be trained should be already initialized
        :param games: games to play while training
        :param update_steps: steps between two updates of the target net, if DoubleDQN is additionally activated
        :param steps_epsilon: steps used to reduce epsilon from 1 in step 0 to EPSILON_END
        :param epsilon_end: minimal value of epsilon-parameter for epsilon-greedy learning
        :param batch_size: number of samples from the replay-buffer to learn from
        :param alpha: learning_rate used to weight the loss when performing the weights update
        :param gamma: discount factor to control influence of the final reward on the computed q-values
        :param n: number of time-steps to be taken into account in n-step bootstrapping
        :param score: score to optimize the trained agent for
        :param lamb: lambda value of the lambda-return methods as alternative to the n-step bootstrapping
        :param look_ahead_search: bool flag to control usage of look-ahead-search to guide the training process
        :param supported_search: control flag to enable search support by an array containing all possible next states
                                 disables reselection of an previously selected action
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param data_name: Name of dataset the agent in trained on
        """
        super(ValueAgentTrainer, self).__init__(
            training_agent=training_agent, games=games, steps_epsilon=steps_epsilon, epsilon_end=epsilon_end,
            alpha=alpha, gamma=gamma, n=n, lamb=lamb, score=score, look_ahead_search=look_ahead_search,
            supported_search=supported_search, refinement=refinement, data_name=data_name)

        self.policy_net = self.training_agent.get_network()
        self.target_net = self.training_agent.get_network()

        # really uncouple the two networks
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.update_steps = update_steps if update_steps != 0 else self.num_seqs * 50
        self.batch_size = batch_size

        self.erb = PrioERB(10000, self.num_seqs)
        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_net.parameters())

    def learn(self, tmp_erb, step=0):
        """
        Performs the learning process.
        :param tmp_erb:
        :param step:
        :returns: the current loss (needed to supervise the training process)
        """
        # fill the experience replay buffer from the actual replay, if possible
        states, actions, rewards, next_states, dones = zip(*tmp_erb)

        # compute the rewards of the episode according to the chosen learning strategy
        if self.n == LAMBDA_LEARN:
            rewards = lambda_rewards(rewards, self.gamma, self.lamb)
        elif self.n == MC_LEARN:
            rewards = [rewards[-1]] * len(states)
            rewards = [self.gamma ** i * rewards[i] for i in reversed(range(len(rewards)))]
        else:
            rewards = self.td_rewards(states, rewards, self.gamma)
        for s, a, r, n_s, d in zip(states, actions, rewards, next_states, dones):
            self.erb.add((s, a, r, n_s, d))

        # learn from experience if there are enough examples to fill a batch
        if len(self.erb) > self.batch_size:
            idxs, batch, _ = self.erb.sample(self.batch_size)  # use for prioritized Experience-Replay
            states, actions, rewards, next_states, dones = zip(*batch)

            # put all necessary information into tensors
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = np.asarray(dones)
            dones[dones is True] = 1
            dones[dones is False] = 0
            dones = torch.LongTensor(dones)

            if self.refinement:
                actions += 1

            # compute loss
            # compute the outcome of the network for the given states according to the selected actions
            curr_q = self.policy_net.forward(states)
            curr_q = curr_q.gather(1, actions.unsqueeze(1)).squeeze()

            # compute the q-value estimation of the following state
            next_q = self.target_net.forward(next_states)

            # Max Q-Value for the upcoming state after performing an action on a state
            max_next_q = next_q.squeeze().max(1)[0]
            expected_q = rewards + dones * self.alpha * max_next_q

            # update weights
            loss = self.loss_function(curr_q, expected_q)
            self.optimizer.zero_grad()  # Update/Learn networks parameters
            loss.backward()
            self.optimizer.step()

            # update the weights/probabilities in the prioritized ERB and also the target-net if used
            self.erb.update(idxs, loss.item())  # Use for prioritized Experience-Replay
            self.update_nets()
            return loss.item()

        # update the target net
        if (step + 1) % self.update_steps == 0:
            self.update_nets()

        return 0

    def update_nets(self):
        """
        Updates the weights of the target net from the policy-net.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":  # this file was run from the command line
    print("######################################")
    print("##Starting training of a value agent##")
    print("######################################")
    print()

    score = SP_SCORE

    seqs = sequences = get_sequences(count=3, length=6, different=True)
    agent = ValueAgent(seqs, network_object=TinyValueFFNN)
    vat = ValueAgentTrainer(agent, n=MC_LEARN)
    vat.run()

    # compute the resulting multiple sequence alignment
    (best_profile, best_permutation), _ = vat.evaluate_training()
    reward = best_profile.score()
    print(str(best_profile))
    print("Score:", reward[score], F"({best_permutation})")

    start = Profile(["ctattg", "ctaccg", "ctatgt"])
    print(start)
    print("Score:", start.score()[score])
    agent = ValueAgent(sequences=start, network_object=TinyValueFFNN, refinement=True)
    tat = ValueAgentTrainer(agent, n=MC_LEARN, refinement=True)
    tat.run(True, False)
    (best_ref_profile, best_ref_permutation), _ = tat.evaluate_training()
    print(str(best_ref_profile))
    reward = best_ref_profile.score()
    print("Score:", reward[score], F"({best_ref_permutation})")
