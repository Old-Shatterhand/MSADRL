import numpy as np

from agent_training.FunctionTrainer import FunctionTrainer
from agents.table_solver import TableSolver
from utils.constants import MC_LEARN, SP_SCORE
from utils.profile import Profile
from utils.utils import lambda_rewards, get_sequences


class TableAgentTrainer(FunctionTrainer):
    def __init__(self, training_agent: TableSolver, games=0, steps_epsilon=0, epsilon_end=0.1, alpha=0.8, gamma=0.9,
                 lamb=0.1, n=0, score=SP_SCORE, look_ahead_search=False, supported_search=False, refinement=False,
                 data_name="SetX"):
        """
        Initialized an agent trainer for tabular agent-training and all needed fields
        :param training_agent: agent that is trained in the training
        :param games: games to play at maximum while training
        :param steps_epsilon: steps used to reduce epsilon from 1 in step 0 to EPSILON_END
        :param epsilon_end: minimal value of epsilon-parameter for epsilon-greedy learning
        :param alpha: learning_rate used to weight the loss when performing the weights update
        :param gamma: discount factor to control influence of the final reward on the computed q-values
        :param n: control for n-step temporal-difference learning or Monte-Carlo algorithms
        :param lamb: lambda value of the lambda-return methods as alternative to the n-step bootstrapping
        :param score: score to optimize the trained agent for
        :param look_ahead_search: bool flag to control usage of look-ahead-search to guide the training process
        :param supported_search: control flag to enable search support by an array containing all possible next states
                                 disables reselection of an previously selected action
        :param refinement: flag indicating the refinement step of a profile to be learned in this training
        :param data_name: Name of dataset the agent in trained on
        """
        super(TableAgentTrainer, self).__init__(
            training_agent=training_agent, games=games, steps_epsilon=steps_epsilon, epsilon_end=epsilon_end,
            alpha=alpha, gamma=gamma, lamb=lamb, n=n, score=score, look_ahead_search=look_ahead_search,
            supported_search=supported_search, refinement=refinement, data_name=data_name)

    def learn(self, replay, step=0):
        """
        learn from the last performed alignment according to the selected learning_type
        :param replay: replay-buffer from last played game
        :param step: step in which this learning method is called (used for epsilon-greediness)
        :return: loss based on changed q-values
        """
        states, actions, rewards, next_states, dones = zip(*replay)

        # compute the reward of the episode according to the learning strategy chosen
        if self.lamb is not None:
            rewards = lambda_rewards(rewards, self.gamma, self.lamb)
        elif self.n is not None:
            if self.n == MC_LEARN:
                rewards = [rewards[-1]] * len(states)
                rewards = [self.gamma ** i * rewards[i] for i in reversed(range(len(rewards)))]
            else:
                rewards = self.td_rewards(states, rewards, self.gamma)

        dones = np.array(dones)
        dones[dones is True] = 1
        dones[dones is False] = 0

        # compute the q-values needed for learning
        q_vals = [self.training_agent.table[state][action] for state, action in zip(states, actions)]
        next_q_vals = [self.training_agent.table[state].argmax(0) for state in next_states]
        new_q_vals = [old_q + self.alpha * (rew + self.gamma * new_q - old_q) for old_q, rew, new_q in
                      zip(q_vals, rewards, next_q_vals)]

        # update the entries in the table
        for state, action, new_q in zip(states, actions, new_q_vals):
            self.training_agent.table[state][action] = new_q

        return sum([abs(new_q - old_q) for new_q, old_q in zip(new_q_vals, q_vals)])

    '''
    def learn_online(self, state, action, reward, next_state, eligibility_trace):
        q_val = self.training_agent.table[state][action]
        next_q_val = self.training_agent.table[next_state].argmax(0)
        delta = reward + next_q_val - q_val
        self.training_agent.table[state][action] += self.alpha * delta
        return delta
    '''


if __name__ == "__main__":  # this file was run from the command line
    print("######################################")
    print("##Starting training of a table agent##")
    print("######################################")
    print()

    score = SP_SCORE

    agent = TableSolver(sequences=get_sequences(count=3, length=6, different=True))
    tat = TableAgentTrainer(agent, alpha=0.8, gamma=0.9, n=MC_LEARN)
    tat.run(True, False)

    # compute the resulting multiple sequence alignment
    (best_profile, best_permutation), _ = tat.evaluate_training()
    reward = best_profile.score()
    print(str(best_profile))
    print("Score:", reward[score], F"({best_permutation})")

    start = Profile(["ctattg", "ctaccg", "ctatgt"])
    print(start)
    print("Score:", start.score()[score])
    agent = TableSolver(sequences=start, refinement=True)
    tat = TableAgentTrainer(agent, alpha=0.8, gamma=0.9, epsilon_end=0.1, n=MC_LEARN, refinement=True)
    tat.run(True, False)
    (best_ref_profile, best_ref_permutation), _ = tat.evaluate_training()
    print(str(best_ref_profile))
    reward = best_ref_profile.score()
    print("Score:", reward[score], F"({best_ref_permutation})")
