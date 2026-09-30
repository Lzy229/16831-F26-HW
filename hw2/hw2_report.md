# hw2
## Code
### train
- the class trainerPG `run_training_loop()`
- call the class RL_trainer `run_training_loop()`, the collect policy is the same as eval policy
- call `train_agent()` for the train and update. # TODO

### train_agent() copy from hw1
- the whole iteration is `self.params['num_agent_train_steps_per_iter']`
- the agent is defined by the `agent_class`, `agent_class` is `PGagent`
- sample the data from replay buffer
    - the agent `sample`, how much data = self.params['train_batch_size']
    - the replay buffer
- train the agent using the data
    - the agent `train`, using the data get from sample

### train
- `observation`, `actions`, `reward` are obtained through sample process. 
- use `calculate_q_vals` to estimates qvals
- use `estimate_estimate_advantage` to compute advantages
- call actor's `update` to update 

#### `reward_list`
- a batch for rewards of n traj

#### `calculate_q_vals`
- $Q(S,A)\leftarrow Q(S,A) + \alpha (R + \gamma Q({S}',{A}') - Q(S,A))$, because its a sampled traj, so there is no transition distribution(the possibility)
- use `_discounted_return` for no reward to go
- use `_discounted_cumsum` for reward to go
- `q_vals` is the average of all q_vals, using monte carlo

##### `_discounted_return`
- array where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
- for no reward to go
    - compute the whole traj for whole array.

##### `_discounted_cumsum`
- returns an array where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
- 

#### `estimate_advantage`
- `obs`, `rewards`, `terminals` are all one-dimension array
- if `self.gae_lambda` is not none, then do the gae
    - use nn.base_line to process value, `base_line` is a mlp, the `run_baseline_prediction` do process 
- if is none, directly use `q_values` and `values` to estimate advantages
- do `normalize` 

### `update`
- use `forward` to get the distribution of actions space
#### `forward`
- input is observation，the output is actions distribution
    - how to get the action? distribution.sample()
- if the action space is discrete(离散的动作)
    - get the distribution (probability) of each action.
- if the action space is not discrete(连续的动作)
    - get the 

### base_line
- use a mlp to calculate values:
    - `values_normalized = self.actor.run_baseline_prediction(obs)`
- the value function will be trained to approximate sum of future rewards starting from a particular state
    - `values = q_values - advantages` 

## debug
### advantages is (0,1)
- check `estimate_advantage`
    - for no baseline, the advantages are the copy of q_values, so the `q_values.shape` is (0, ), go check `calculate_q_vals`
- check `calculate_q_vals`
    - use `concatenate` for append

### ValueError: Value is not broadcastable with batch_shape+event_shape: torch.Size([1502]) vs torch.Size([1503])
- the shape of `actions_distribution.log_prob(actions[i:])` != the shape of `advantages[i:]`

### error 3
- sum() received an invalid combination of arguments - got (out=NoneType, axis=int, ), but expected one of: * (*, torch.dtype dtype)
    - `np.sum` receive the tensor, use `torch.sum`

## experiment
### Experiment 1
#### Configurations
- hw2/data/q1_sb_no_rtg_dsa_CartPole-v0_27-09-2026_21-14-29
- hw2/data/q1_sb_rtg_dsa_CartPole-v0_27-09-2026_21-15-09
- hw2/data/q1_sb_rtg_na_CartPole-v0_27-09-2026_21-18-01
- hw2/data/q1_lb_no_rtg_dsa_CartPole-v0_27-09-2026_21-19-38
- hw2/data/q1_lb_rtg_dsa_CartPole-v0_27-09-2026_21-20-43
- hw2/data/q1_lb_rtg_na_CartPole-v0_27-09-2026_21-21-53

#### plot
- do plot using python 
    - small batch
    - large batch

The best configuration of CartPole in both the large and small batch cases should converge to a maximum
score of 200.

### Experiment 2
- `batch_size = 2000, learning_rate = 8e-3`
- `batch_size = 3100, learning_rate = 1e-2`

### Experiment 3
- 