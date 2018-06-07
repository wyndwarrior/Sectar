class Sampler:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy