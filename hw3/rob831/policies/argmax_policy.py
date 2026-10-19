import numpy as np
from rob831.infrastructure import pytorch_util as ptu

class ArgMaxPolicy(object):

    def __init__(self, critic):
        self.critic = critic

    def get_action(self, obs):
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]
        
        ## TODO return the action that maximizes the Q-value 
        # at the current observation as the output

        observation = ptu.from_numpy(observation)
        qa_value = self.critic.q_net(observation)
        qa_value = ptu.to_numpy(qa_value)

        action = np.argmax(qa_value, axis=1)

        return action.squeeze()
