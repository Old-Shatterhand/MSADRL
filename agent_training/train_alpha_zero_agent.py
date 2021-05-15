import torch
from torch import optim, nn

from agents.alpha_zero_agent import AlphaZeroAgent
from agents.mcts_agent import MCTSAgent
from networks.alphazero_networks import TinyA0_Network
from utils.agent_trainer import AgentTrainer
from utils.constants import SP_SCORE
from utils.profile import Profile
from utils.utils import get_sequences, linearize_state


class AlphaZeroAgentTrainer(AgentTrainer):
    def __init__(self, training_agent, games=0, simulations=10, rollouts=1, c=1, score=SP_SCORE, refinement=False,
                 adjust=True, data_name='SetX'):
        """
        Trainer for AlphaZero agents that shall solve the multiple sequence alignment problem
        :param training_agent: instance of the AlphaZero Solver
        :param games: number of games to play during training
        :param simulations: number of mcts-simulations to perform in each state
        :param rollouts: number of rollouts to perform per simulation
        :param c: exploration-exploitation-balancing hyperparameter to fine-tune the alignments
        :param score: score to optimize the agent for
        :param refinement: flag indicating the type of optimization
        :param data_name: name of the dataset training on
        """
        super(AlphaZeroAgentTrainer, self).__init__(
            training_agent=training_agent, games=games, score=score, refinement=refinement, data_name=data_name)

        # initialize the mcts-agent that is then used to generate the training-data
        self.mcts_generator = MCTSAgent(self.training_agent.profile if self.refinement else self.sequences,
                                        simulations=simulations, rollouts=rollouts, c=c, refinement=refinement,
                                        adjust=adjust)

        # initialize some parameters for training
        self.optimizer = optim.Adam(self.training_agent.net.parameters())
        self.value_loss_function = nn.MSELoss()
        self.score = score

    def set_align_table(self, align_table):
        """
        Overrides the method from the super-class, since this trainer is not alone interacting with an environment
        :param align_table: align-table instance to store alignments in and to get them from
        """
        super().set_align_table(align_table)
        self.mcts_generator.set_align_table(align_table)

    def get_align_table(self):
        """
        Overrides the method from the super-class, since this trainer is not alone interacting with an environment
        Here, it's sufficient to return the MCTS-Agents' table since this contains exactly the same or even more states
        than the table from the own environment
        (proof by construction of this trainer and the usage in the "menerate-mcts-episodes"-method)
        :return: align-table used and extended in this algorithm
        """
        return self.mcts_generator.get_align_table()

    def train(self, print_progress):
        """
        Perform the main part of the training
        :param print_progress: flag indicating to print the progress to the commandline
        :return: the losses and the number of fails in the test steps
        """
        episode_loss = [0, 0]
        avg_rewards, avg_losses, avg_fails = [], [], []

        for step in range(self.games):
            if (step + 1) % self.plot_size == 0 or self.env.align_table.is_full():
                # test the progress the agent made by performing 10 alignments and evaluating the results
                tmp = [self.env.reset() for _ in range(10)]
                episode_reward = sum([s[self.score] for s, _, _, _ in tmp]) * 10
                episode_fails = sum([not d for _, _, _, d in tmp]) * 10

                # print the progress
                tmp_reward, tmp_loss, tmp_fails = self.print_progress(print_progress, step, episode_reward,
                                                                      episode_loss, episode_fails)
                avg_rewards.append(tmp_reward)
                avg_losses.append(tmp_loss)
                avg_fails.append(tmp_fails)
                episode_loss = [0, 0]

                # if all alignments have been found exit
                if self.env.align_table.is_full():
                    if print_progress:
                        print("Search exited. All alignments have been visited and optimality is guaranteed.")
                    break

            # generate the training-batch for this step and learn from it
            episode_loss = [e + l for e, l in zip(episode_loss, self.learn(self.generate_mcts_episode()))]

        return avg_rewards, avg_losses, avg_fails

    def learn(self, tmp_erb, step=0):
        """
        Learn from an self-play using, MCTS-supervised batch
        :param tmp_erb: data of the batch to train on
        :param step: step indicating the number of how often the agent has learned before
        :return: loss of the value-function and the policy approximation
        """
        # extract the data from the training batch
        states = torch.Tensor([s for s, _, _ in tmp_erb])
        probs = torch.Tensor([p for _, p, _ in tmp_erb])
        s_ests = torch.Tensor([e for _, _, e in tmp_erb])

        # compute what the net thinks about the data
        net_outputs = self.training_agent.net.forward(states)
        net_ests = net_outputs[0]
        net_probs = net_outputs[1]

        # compute the individual losses
        value_loss = self.value_loss_function(s_ests, net_ests)
        policy_loss = torch.diagonal(torch.matmul(-probs, net_probs.log().T)).sum()
        loss = value_loss + policy_loss
        loss = loss.sum()

        # apply these losses to the
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return value_loss.item(), policy_loss.item()

    def generate_mcts_episode(self):
        """
        Generate training data based on which actions the network would perform and
        what an MCTS-Supervisor thinks about the resulting states
        :return:
        """
        replay_buffer = []
        _, state, _, done = self.env.soft_reset()

        while not done:
            # evaluate the state using mcts to get move-probabilities and a state estimate
            probs, s_est = self.mcts_generator.get_probabilities(state)

            # use the networks actual state to select the next action
            with torch.no_grad():
                action = self.training_agent.select(state)

            # append the data to the replay-buffer
            replay_buffer.append((linearize_state(state, self.num_seqs), probs, s_est))

            # and apply the selected action to the state
            _, state, _, done = self.env.step(action)
        return replay_buffer


if __name__ == "__main__":  # this file was run from the command line
    print("#############################################")
    print("##Starting training of a actor-critic agent##")
    print("#############################################")
    print()

    score = SP_SCORE

    agent = AlphaZeroAgent(sequences=get_sequences(count=3, length=6, different=True),
                           network_object=TinyA0_Network)
    a0t = AlphaZeroAgentTrainer(training_agent=agent, simulations=50, adjust=True)
    a0t.run(progress_print=True)

    # compute the resulting multiple sequence alignment
    (best_profile, best_permutation), _ = a0t.evaluate_training()
    reward = best_profile.score()
    print(str(best_profile))
    print("Score:", reward[score], F"({best_permutation})")

    start = Profile(["ctattg", "ctaccg", "ctatgt"])
    print(start)
    print("Score:", start.score()[score])
    agent = AlphaZeroAgent(sequences=start, network_object=TinyA0_Network, refinement=True, adjust=True)
    a0t = AlphaZeroAgentTrainer(agent, simulations=50, refinement=True)
    a0t.run(True, False)
    (best_ref_profile, best_ref_permutation), _ = a0t.evaluate_training()
    print(str(best_ref_profile))
    reward = best_ref_profile.score()
    print("Score:", reward[score], F"({best_ref_permutation})")
