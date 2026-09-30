import numpy as np

from rob831.agents.base_agent import BaseAgent
from rob831.policies.MLP_policy import MLPPolicyPG
from rob831.infrastructure.replay_buffer import ReplayBuffer

from rob831.infrastructure.utils import normalize, unnormalize

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super().__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma'] # discount factor 
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # TODO: update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        q_vals = self.calculate_q_vals(rewards_list)
        advantages = self.estimate_advantage(observations, rewards_list, q_vals, terminals)

        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log

        train_log = self.actor.update(observations, actions, advantages, q_vals)

        # raise NotImplementedError

        return train_log


    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # TODO: return the estimated qvals based on the given rewards, using
            # either the full trajectory-based estimator or the reward-to-go
            # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
            # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
            # self._discounted_cumsum (you will need to implement these). These
            # functions should only take in a single list for a single trajectory.

        # Case 1: trajectory-based PG
        # Estimate Q^{pi}(s_t, a_t) by the total discounted reward summed over entire trajectory
        # HINT3: q_values should be a 1D numpy array where the indices correspond to the same
        # ordering as observations, actions, etc.

        q_values = np.array([])

        if not self.reward_to_go:
            #use the whole traj for each timestep
            for i in range(len(rewards_list)):
                
                q_values = np.concatenate([q_values, self._discounted_return(rewards_list[i])])

            # raise NotImplementedError

        # Case 2: reward-to-go PG
        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            for i in range(len(rewards_list)):
                q_values = np.concatenate([q_values, self._discounted_cumsum(rewards_list[i])])

            # raise NotImplementedError
        # print(q_values.shape)
        return q_values  # return an array

    def estimate_advantage(self, obs, rewards_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:

            values_normalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_normalized.ndim == q_values.ndim
            ## TODO: values were trained with standardized q_values, so ensure
                ## that the predictions have the same mean and standard deviation as
                ## the current batch of q_values


            # raise NotImplementedError
            values = unnormalize(values_normalized, np.mean(q_values), np.std(q_values))

            if self.gae_lambda is not None:
                ## append a dummy T+1 value for simpler recursive calculation
                print("gae_lambda is not none")
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rewards = np.concatenate(rewards_list) # 

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## TODO: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula
                    if terminals[i] != 1:
                        next_value = values[i+1]
                    else:
                        next_value = 0.0

                    delta = rewards[i] + self.gamma * next_value - values[i]

                    advantages[i] = delta + self.gamma * self.gae_lambda * advantages[i+1]

                    
                    # raise NotImplementedError

                # remove dummy advantage
                advantages = advantages[:-1]

            else:
                ## TODO: compute advantage estimates using q_values, and values as baselines
                # raise NotImplementedError
                advantages = q_values - values
                # the definition of advantages

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()
            # print(advantages.shape)

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one

            # raise NotImplementedError
            advantages = normalize(advantages, np.mean(advantages), np.std(advantages))

        # print(advantages.shape)
        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create discounted_returns
        q_vals = 0
        for i in range(len(rewards)):
            q_vals += pow(self.gamma, i) * rewards[i]
        # raise NotImplementedError

        discounted_returns = np.ones_like(rewards) * q_vals
        return discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `discounted_cumsums`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine

        # from scipy.linalg import toeplitz

        # q_vals = np.ones_like(rewards) * self.gamma
        # pow = np.arange(len(rewards))
        # vector = np.power(q_vals, pow)

        # first_col = [vector[0]] + [0] * (len(rewards) - 1)
        # matrix = toeplitz(first_col, vector)

        # discounted_cumsums = np.dot(matrix, rewards.T)

        lenth = len(rewards)
        discounted_cumsums = np.zeros(lenth + 1)
        for i in reversed(range(len(rewards))):
            discounted_cumsums[i] = rewards[i] + self.gamma * discounted_cumsums[i+1]

        # raise NotImplementedError

        return discounted_cumsums[: -1]
