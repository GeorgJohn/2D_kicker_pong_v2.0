class EpisodeHistory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.estimated_value = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []
        self.learning_value_pg = []

    def add_to_history(self, state, action, value, reward, state_prime):
        self.states.append(state)
        self.actions.append(action)
        self.estimated_value.append(value)
        self.rewards.append(reward)
        self.state_primes.append(state_prime)


class Memory(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.estimated_value = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []
        self.learning_value_pg = []

    def reset_memory(self):
        self.states = []
        self.actions = []
        self.estimated_value = []
        self.rewards = []
        self.state_primes = []
        self.discounted_returns = []
        self.learning_value_pg = []

    def add_episode(self, episode):
        self.states += episode.states
        self.actions += episode.actions
        self.estimated_value += episode.estimated_value
        self.rewards += episode.rewards
        self.discounted_returns += episode.discounted_returns
        self.learning_value_pg += episode.learning_value_pg


class RewardsMemory(object):

    def __init__(self):
        self.rewards = []
        self.values = []
        self.episodes = []
