import numpy as np


class PrioritizedExperienceReplayBuffer:
    def __init__(self, size, input_size):
        """
        ReplayBuffer to select samples from experience, used in DQN-algorithms
        additional feature is the prioritizing of the features, so more interesting samples are selected more often
        :param size: number of elements to store inside this buffer
        :param input_size: length of the arrays inside the buffer
        """
        self.size = size
        self.buffer = np.asarray([([1] * input_size, 0, 0, [1] * input_size, False)], dtype=object)
        self.loss = np.array([0.0])

    def __len__(self):
        """
        Define the length of the buffer as the number of elements stored inside
        :return: length of buffer as defined
        """
        return len(self.buffer)

    def sample(self, amount):
        """
        Samples among buffer.
        :param amount: amount of sampling
        :returns: indices, experiences and priority-values of the samples
        """
        # since the priorities has to be a probability distribution with sum 1 we have to normalize the loss array
        priorities = np.true_divide(self.loss, sum(self.loss))
        if amount > len(self.buffer):
            print("Warning: sample amount > length of buffer. Sampling with replacement!")
            replace = True
        else:
            replace = False
        indices = np.random.choice(len(self.buffer), amount, replace=replace, p=priorities)
        return indices, [self.buffer[i] for i in indices], [priorities[i] for i in indices]

    def add(self, data):
        """
        Appends data to buffer. data con b single entries or a list of entries
        :param data: data to append
        """
        # add list of data ...
        if isinstance(data, list):
            # if necessary, free the space and add the data
            if len(data) + len(self.buffer) >= self.size:
                self.remove(len(data) + len(self.buffer) - self.size)
            for d in data:
                self.add(d)

        # ... or just a single point
        else:
            # if necessary free the space and add the data
            if len(self.buffer) >= self.size:
                self.remove()
            self.buffer = np.append(self.buffer, [np.array(data, dtype=object)], axis=0)
            self.loss = np.append(self.loss, 1)

    def remove(self, count=1):
        """
        remove elements from the buffer in order to create some space for more important entries
        :param count: specification how many entries should be removed
        """
        mask = np.argpartition(self.loss, count)[:count]
        np.delete(self.buffer, mask, axis=0)
        np.delete(self.loss, mask, axis=0)

    def update(self, indices, loss):
        """
        Updates buffer for indices.
        """
        for i in indices:
            self.loss[i] = loss
