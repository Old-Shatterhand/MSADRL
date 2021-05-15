import torch
from torch import optim, nn

from agent_training.PolicyTrainer import PolicyTrainer
from agents.policy_agent import PolicyAgent
from networks.reinforce_networks import TinyREINFORCENetwork
from utils.constants import SP_SCORE
from utils.profile import Profile
from utils.utils import get_sequences


class PolicyAgentTrainer(PolicyTrainer):
    def __init__(self, training_agent: PolicyAgent, games=0, steps_epsilon=0, epsilon_end=0.1, value_alpha=0.001,
                 policy_alpha=0.001, value_gamma=0.99, policy_gamma=0.99, baseline=False, score=SP_SCORE,
                 look_ahead_search=False, supported_search=False, refinement=False, data_name="SetX"):
        """
        Initializing the trainer for the policy agent with all needed fields
        to perform training following the REINFORCE algorithm
        :param training_agent: initialized instance of the trained policy agent
        :param games: games to play during training
        :param steps_epsilon: count how many steps to perform until epsilon shall reach epsilon end
        :param epsilon_end: final value of epsilon at the end og epsilon decreasing
        :param value_alpha: learning_rate used to weight the loss when performing the weights update for value-approx.
        :param policy_alpha: learning_rate used to weight the loss when performing the weights update for policy-approx.
        :param value_gamma: discount factor to control influence of future reward on weights update for value-approx.
        :param policy_gamma: discount factor to control influence of future reward on weights update for policy-approx.
        :param baseline: flag indicating use of baseline for REINFORCE algorithm of not
        :param score: score to optimize the trained agent for
        :param look_ahead_search: bool flag to control usage of look-ahead-search to guide the training process
        :param supported_search: control flag to enable search support by an array containing all possible next states
                                 disables reselection of an previously selected action
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param data_name: Name of dataset the agent in trained on
        """
        super(PolicyAgentTrainer, self).__init__(
            training_agent=training_agent, games=games, steps_epsilon=steps_epsilon, epsilon_end=epsilon_end,
            value_alpha=value_alpha, policy_alpha=policy_alpha, value_gamma=value_gamma, policy_gamma=policy_gamma,
            score=score, look_ahead_search=look_ahead_search, supported_search=supported_search, refinement=refinement,
            data_name=data_name)

        self.baseline = baseline

        self.policy_optimizer = optim.Adam(self.training_agent.net.policy_net.parameters())
        if self.baseline:
            self.value_optimizer = optim.Adam(self.training_agent.net.value_net.parameters())
            self.value_loss_function = nn.MSELoss()

    def learn(self, tmp_erb, step=0):
        """
        Learn from the last played game and fit the weights according to a computed gradient
        :param tmp_erb: replay buffer from last played game
        :param step: not used in this implementation
        :return: loss, needed for optimizations
        """
        states, actions, rewards, next_states, dones, values, log_probs = zip(*tmp_erb)
        states = torch.FloatTensor(states)
        discounted_rewards = []

        # compute the discounted rewards of this episode...
        for t in range(len(rewards)):
            G, c = 0, 0
            for r in rewards[t:]:
                G += self.value_gamma ** c * r
                c += 1
            discounted_rewards.append(G)

        # ... and normalize them
        discounted_rewards = torch.tensor(discounted_rewards)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # compute the policy gradient
        policy_gradient = []

        if self.baseline:
            # compute the state-values from the baseline network
            curr_q = self.training_agent.net.value_net.forward(states).squeeze()
            delta = discounted_rewards - curr_q
            loss = self.value_loss_function(curr_q, discounted_rewards)

            # fill the policy_gradient according to the state-values
            for log_prob, d in zip(log_probs, delta):
                policy_gradient.append(self.value_alpha * log_prob * d.item())

            # apply the value gradient to the value network
            self.value_optimizer.zero_grad()
            value_loss = loss.item()
            loss.backward(retain_graph=True)
            self.value_optimizer.step()
        else:
            for log_prob, G in zip(log_probs, discounted_rewards):
                policy_gradient.append(self.policy_alpha * log_prob * G)
            value_loss = 0

        # apply the policy gradient to the policy network
        policy_gradient = -torch.stack(policy_gradient).sum()
        policy_loss = policy_gradient.item()
        self.policy_optimizer.zero_grad()
        policy_gradient.backward()
        self.policy_optimizer.step()

        # return the loss for optimization and learning control
        return policy_loss, value_loss


if __name__ == "__main__":  # this file was run from the command line
    print("#######################################")
    print("##Starting training of a policy agent##")
    print("#######################################")
    print()

    score = SP_SCORE

    agent = PolicyAgent(sequences=get_sequences(count=3, length=6, different=True),
                        network_object=TinyREINFORCENetwork)
    pat = PolicyAgentTrainer(agent, value_gamma=0.99, value_alpha=0.8, baseline=True)
    pat.run()

    # compute the resulting multiple sequence alignment
    (best_profile, best_permutation), _ = pat.evaluate_training()
    reward = best_profile.score()
    print(str(best_profile))
    print("Score:", reward[score], F"({best_permutation})")

    start = Profile(["ctattg", "ctaccg", "ctatgt"])
    print(start)
    print("Score:", start.score()[score])
    agent = PolicyAgent(sequences=start, network_object=TinyREINFORCENetwork, refinement=True)
    tat = PolicyAgentTrainer(agent, value_alpha=0.01, value_gamma=0.9, epsilon_end=0.1, refinement=True)
    tat.run(True, False)
    (best_ref_profile, best_ref_permutation), _ = tat.evaluate_training()
    print(str(best_ref_profile))
    reward = best_ref_profile.score()
    print("Score:", reward[score], F"({best_permutation})")
