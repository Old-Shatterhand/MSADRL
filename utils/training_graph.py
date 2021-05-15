import matplotlib.pyplot as plt


def make_patch_spines_invisible(ax):
    """
    Set the third y-axis of a plot away from the second to that they do not overlap
    :param ax: axis to be moves
    """
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


class TrainingGraph:
    def __init__(self):
        """
        Class to visualize the training-progress of an Agent
        Supports two lines in the graph, one for training and one for validation-progress
        Use for debug-proposes since it does not support saving the image
        """
        self.points = {key: [False, []] for key in ["Reward", "Loss", "PolicyLoss", "ValueLoss", "Fails"]}
        self.colors = {"Reward": "green", "Loss": "blue", "PolicyLoss": "navy", "ValueLoss": "deepskyblue",
                       "Fails": "red"}

    def add(self, label, score):
        """
        Add new data-point
        :param label: label of the according curve
        :param score: value of the points
        """
        self.points[label][1].append(score)
        self.points[label][0] |= True

    def create_graph(self, title, max_reward):
        """
        Create an graph with extended information on the training process
        This graph can contain (for each episode) the reward, two losses (one for the value-agent and one for policy-
        agents (used in baseline REINFORCEMENT and ActorCritic implementations)
        and the ratio of selected invalid sequences to be aligned next
        :param title: title of the plot
        :param max_reward: reward for best alignment found during the training upper bound for the reward axis
        """
        fig, ax1 = plt.subplots()
        plt.title(title)
        lines = []

        # fill reward-axis and the x-axis
        ax1.set_xlabel("Number of episodes (*100)")
        ax1.set_ylabel("Column-Score (EM/AL)", color=self.colors["Reward"])
        ax1.set_ylim(ymin=0, ymax=max_reward * 1.05)
        lines.append(ax1.plot(self.points["Reward"][1], color=self.colors["Reward"], label="Reward")[0])
        ax1.tick_params(axis='y', labelcolor=self.colors["Reward"])

        # draw the loss axis if at least one loss is given
        if self.points["Loss"][0] or self.points["PolicyLoss"][0] or self.points["ValueLoss"][0]:
            # create additional y-axis
            ax2 = ax1.twinx()
            ax2.set_ylabel("Episode-Loss", color=self.colors["Loss"])
            ax2.tick_params(axis='y', labelcolor=self.colors["Loss"])

            # normal loss (mostly used in function-agents)
            if self.points["Loss"][0]:
                lines.append(ax2.plot(self.points["Loss"][1], color=self.colors["Loss"], label="Loss")[0])

            # loss of policy approximations
            if self.points["PolicyLoss"][0]:
                lines.append(
                    ax2.plot(self.points["PolicyLoss"][1], color=self.colors["PolicyLoss"], label="Policy-Loss")[0])

            # loss of value approximations in REINFORCE and ActorCritics implementations
            if self.points["ValueLoss"][0]:
                lines.append(
                    ax2.plot(self.points["ValueLoss"][1], color=self.colors["ValueLoss"], label="Value-Loss")[0])

        # if given, draw another additional y-axis for the fails
        if self.points["Fails"][0]:
            # create third y-axis and shift to not overlap second y-axis
            ax3 = ax1.twinx()
            ax3.spines["right"].set_position(("axes", 1.2))
            make_patch_spines_invisible(ax3)
            ax3.spines["right"].set_visible(True)

            # plot the invalid action ratio
            ax3.set_ylabel("imp. action ratio", color=self.colors["Fails"])
            lines.append(ax3.plot(self.points["Fails"][1], color=self.colors["Fails"], label="Fails")[0])
            ax3.tick_params(axis='y', labelcolor=self.colors["Fails"])

        # fig.tight_layout()
        plt.legend(lines, [l.get_label() for l in lines])
        plt.show()
