import time

import torch

from utils.alignment import align_progressive
from utils.constants import MC_LEARN, SP_SCORE
from utils.training_graph import TrainingGraph
from utils.wrapper import AlignmentWrapper, RefinementWrapper, Profile


class AgentTrainer:
    def __init__(self, training_agent, games=0, steps_epsilon=0, epsilon_end=0.1, n=MC_LEARN, score=SP_SCORE,
                 look_ahead_search=False, supported_search=False, data_name='SetX', refinement=False):
        """
        Filling needed fields by extracting those from the agent
        :param training_agent: the agent to be trained should be already initialized
        :param games: games to play while training
        :param steps_epsilon: steps used to reduce epsilon from 1 in step 0 to EPSILON_END
        :param epsilon_end: minimal value of epsilon-parameter for epsilon-greedy learning
        :param n: control for n-step temporal-difference learning or Monte-Carlo algorithms
        :param look_ahead_search: bool flag to control usage of look-ahead-search to guide the training process
        :param supported_search: control flag to enable search support by an array containing all possible next states
                                 disables reselection of an previously selected action
        :param data_name: Name of dataset the agent in trained on
        """
        # parameterize the learning process
        self.num_seqs = len(training_agent.get_sequences()) + (1 if refinement else 0)
        self.games = games if games != 0 else self.num_seqs * 500
        self.plot_size = 100
        self.steps_epsilon = steps_epsilon if steps_epsilon != 0 else self.num_seqs * 50
        self.epsilon_end = epsilon_end
        self.data_name = data_name
        self.n = min(n, self.num_seqs)
        self.score = score
        self.look_ahead_search = look_ahead_search and not refinement
        self.supported_search = supported_search and not refinement
        self.refinement = refinement

        # initialize properties from the agent
        self.training_agent = training_agent
        self.input_size = training_agent.get_input_size()
        self.sequences = training_agent.get_sequences()
        self.network_path = training_agent.get_network_path()
        self.net = training_agent.get_network()

        # other utils needed to train
        if self.refinement:
            self.env = RefinementWrapper(Profile(self.sequences), self.training_agent, self.score)
        else:
            self.env = AlignmentWrapper(self.sequences, self.training_agent, self.score)
        self.graph = TrainingGraph()

    def run(self, progress_print=True, graph_print=False):
        """
        Runs the training
        :param progress_print: bool-flag to enable the live-prints of the average reward during the training
        :param graph_print: flag to control the creation and showing of a graph of the training process
        """
        # run the training and track the time needed to train
        start = time.time()
        rewards, losses, fails = self.train(progress_print)
        end = time.time()

        # save the agent and plot the statistics of this run
        if graph_print:
            self.graph.create_graph(F"Training of {self.training_agent.name()} on {self.data_name}",
                                    self.env.best_alignment[0].score()[self.score])
        self.save_agent()
        if progress_print:
            print("Trainer ran for %.2f seconds" % (end - start))
        return rewards, losses, fails, (end - start)

    def td_rewards(self, states, rewards, gamma):
        """
        Compute the temporal-difference rewards for the agent training
        :param states: states of the last episode
        :param rewards: rewards per steps of the last episode
        :param gamma: discount factor
        :return: return the accumulated and discounted rewards of the last episode to use in TD-learning
        """
        rewards = list(rewards)
        n_step_rewards = []

        # compute the discounted weighting vector for the n
        discounts = [gamma ** i for i in range(1, self.n + 1)]
        for i in range(len(rewards)):
            # compute the accumulation of rewards
            if self.n > 0:
                reward = sum([rewards[t] * discounts[t - i] for t in range(i, min(len(rewards), self.n + i))])
            else:
                reward = rewards[i]

            # if necessary add an estimate for the last upcoming state (after n steps) to the reward
            if i + self.n < len(rewards):
                reward += self.training_agent.get_state_estimate(states[i + self.n])

            n_step_rewards.append(reward)

        return n_step_rewards

    def train(self, print_progress):
        """
        Performs the learning process. Overridden in Function- and PolicyTrainer
        :return: the returns, losses and invalid action ratios computed during the training process
            (usable for analytical and optimization tasks)
        """
        return [], [], []

    def learn(self, tmp_erb, step=0):
        """
        Just an abstract method to be callable from this class that has to be implemented in the extending classes
        :param tmp_erb: replay-buffer generated in last episode
        :param step: number of episodes played yet
        :return: loss calculated in this method, can be used for optimization tasks
        """
        pass

    def print_progress(self, progress_print, step, episode_reward, episode_loss=None, episode_fails=None):
        """
        print the learning progress of a trained agent
        :param progress_print: flag indicating to print the progress
        :param step: steps done until now
        :param episode_reward: accumulated reward since the last print
        :param episode_loss: loss of an agent in the last episodes
        :param episode_fails: number of failed alignments in the last episodes
        :return:
        """
        avg_reward = episode_reward / self.plot_size
        if progress_print:
            print("Step ", step + 1, " of ", self.games)
            print("\tavg. align. score:", avg_reward)
        self.graph.add("Reward", avg_reward)

        # create measurements to be returned in case, that they are not recomputed
        avg_loss = 0
        avg_fails = 0

        if episode_loss is not None:
            if isinstance(episode_loss, tuple) or isinstance(episode_loss, list):
                avg_loss = [e / self.plot_size for e in episode_loss]
                self.graph.add("PolicyLoss", avg_loss[0])
                self.graph.add("ValueLoss", avg_loss[1])
                if progress_print:
                    print("\tavg. policy-loss :", avg_loss[0])
                    print("\tavg. value-loss  :", avg_loss[1])
            else:
                avg_loss = episode_loss / self.plot_size
                self.graph.add("Loss", avg_loss)
                if progress_print:
                    print("\tavg. total loss  :", avg_loss)
        if episode_fails is not None:
            avg_fails = episode_fails / self.plot_size
            self.graph.add("Fails", avg_fails)
            if progress_print:
                print("\tavg. imp. actions:", avg_fails)
        return avg_reward, avg_loss, avg_fails

    def look_ahead(self):
        """
        Look-ahead step in the current state of the environment to select the real best next action
        :return: best action to take in this state
        """
        options = [
            align_progressive(self.env.permutation + [i], self.sequences, self.env.align_table).score()[self.score]
            if v == 1 else 0 for i, v in enumerate(self.env.available)]
        return options.index(max(options))

    def evaluate_training(self):
        """
        Get the best alignment the agents has ever found in its training
        :return: best alignment the environment has found during training
        """
        return self.env.evaluate_training()

    def save_agent(self):
        """
        save the network of the agent in the specified file
        """
        if self.training_agent.load:
            torch.save(self.net.state_dict(), self.network_path)

    def load_agent(self):
        """
        load the weights for the network from the file specified by the agent
        """
        if self.training_agent.load:
            self.net.load_state_dict(torch.load(self.network_path))

    def set_align_table(self, align_table):
        """
        Set the hash-table of previously computed alignments
        :param align_table: table to assign to own table
        """
        self.env.align_table = align_table

    def get_align_table(self):
        """
        Return the hash-table of previously computed alignments
        :return: alignment table computed/extended while computation
        """
        return self.env.align_table
