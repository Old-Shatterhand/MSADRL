import random

import numpy as np

from utils.agent_trainer import AgentTrainer
from utils.constants import MC_LEARN, SP_SCORE
from utils.profile import Profile
from utils.utils import linearize_state


printing = False


class FunctionTrainer(AgentTrainer):
    def __init__(self, training_agent, games=0, steps_epsilon=0, epsilon_end=0, alpha=0.001, gamma=0.9, lamb=0.1,
                 n=MC_LEARN, score=SP_SCORE, look_ahead_search=False, supported_search=False, refinement=False,
                 data_name='SetX'):
        """
        Container class holding methods used in value-function approximating agents like table and value agents
        Filling needed fields by extracting those from the agent
        :param training_agent: the agent to be trained should be already initialized
        :param games: games to play while training
        :param steps_epsilon: steps used to reduce epsilon from 1 in step 0 to EPSILON_END
        :param epsilon_end: minimal value of epsilon-parameter for epsilon-greedy learning
        :param alpha: learning_rate used to weight the loss when performing the weights update
        :param gamma: discount factor to control influence of the final reward on the computed q-values
        :param lamb: lambda value of the lambda-return methods as alternative to the n-step bootstrapping
        :param n: control for n-step temporal-difference learning or Monte-Carlo algorithms
        :param score: score to optimize the agent for
        :param look_ahead_search: bool flag to control usage of look-ahead-search to guide the training process
        :param supported_search: control flag to enable search support by an array containing all possible next states
                                 disables reselection of an previously selected action
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param data_name: Name of dataset the agent in trained on
        """
        super(FunctionTrainer, self).__init__(
            training_agent=training_agent, games=games, steps_epsilon=steps_epsilon, epsilon_end=epsilon_end, n=n,
            score=score, look_ahead_search=look_ahead_search, supported_search=supported_search, refinement=refinement,
            data_name=data_name)

        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb

    def act(self, state, available=None, step=0):
        """
        Selects an action based on the given input-state.
        :param state: state to select action for, must be provided in non-linearized, permutation form
        :param available: vector containing the actions that lead to non-zero reward in this step in one-hot encoding
        :param step: step in which this action is made, needed for epsilon-greediness
        :returns: the selected action
        """
        # Compute the value for epsilon depending on the step
        if step < self.steps_epsilon:
            epsilon = 1 - (1 - self.epsilon_end) * (step / self.steps_epsilon)
        else:
            epsilon = self.epsilon_end
        # depending on the epsilon value, select an action...
        if epsilon > 0 and random.uniform(0, 1) < epsilon:
            if self.look_ahead_search and random.uniform(0, 1) < 0.5:
                action = self.look_ahead()
            else:
                # ...randomly
                action = random.choice(list(set(range(self.num_seqs)) - set(state))) - (1 if self.refinement else 0)
        else:
            # ...depending on the network state
            action = self.training_agent.act(linearize_state(state, self.num_seqs),
                                             available if self.supported_search else None)
        return action

    def train(self, print_progress):
        """
        Performs the learning process.
        :return: the returns, losses and invalid action ratios computed during the training process
            (usable for analytical and optimization tasks)
        """
        episode_reward, episode_loss, episode_fails = 0, 0, 0
        avg_rewards, avg_losses, avg_fails = [], [], []

        # play the games in the training sample selected from randomly samples game-states
        for step in range(self.games):
            # print the progress the model made while learning
            if (step + 1) % self.plot_size == 0 or self.env.align_table.is_full():
                tmp_reward, tmp_loss, tmp_fail = self.print_progress(print_progress, step, episode_reward, episode_loss,
                                                                     episode_fails)
                avg_rewards.append(tmp_reward)
                avg_losses.append(tmp_loss)
                avg_fails.append(tmp_fail)
                episode_reward, episode_loss, episode_fails = 0, 0, 0

                # if all alignments have been found exit
                if self.env.align_table.is_full():
                    if self.env.best_alignment == (Profile([]), None):
                        self.env.best_alignment = self.env.align_table.get_best(self.score)
                    if print_progress:
                        print("Search exited. All alignments have been visited and optimality is guaranteed.")
                    break

            game_reward, state, profile, done = self.env.soft_reset()

            # play new game
            tmp_erb = []
            while not done:
                # compute the action, perform it in the environment and add all stats to the local replay-buffer
                action = self.act(state, available=self.env.available, step=step)
                prev_state = np.array(state)
                game_reward, state, profile, done = self.env.step(action)
                tmp_erb.append((linearize_state(prev_state, self.num_seqs), action, game_reward[self.score],
                                linearize_state(state, self.num_seqs), done))

            # update reward according to the received reward
            episode_reward += game_reward[self.score]

            if not self.refinement and len(state) != self.num_seqs:
                episode_fails += 1

            # learn from replay
            episode_loss += self.learn(tmp_erb, step)

        return avg_rewards, avg_losses, avg_fails
