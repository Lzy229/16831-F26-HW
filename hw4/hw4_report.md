# HW4 Code
## Problem one
### Run
- use `MB_Trainer`, and call `run_training_loop()`
- call `RL_Trainer`, and call `run_training_loop()`, collect policy is `agent.actor`
- 

### `mb_agent` 
- the actor is `MBCPolicy`

#### Train()
- the default `ensemble_size` is 3
    - the `dyn_models` (dynamic models) has `ensemble_size` of `FF_models`
- in one training process, we need to train all the `FF_models` using different batch of `observation`
    - batch size is `num_data_per_ens`
- we are not using `re_n` and `terminal_n`

### `FF_Models`
#### `forward`
- parameters:
    - param obs_unnormalized: Unnormalized observations
    - param acs_unnormalized: Unnormalized actions
    - param obs_mean: Mean of observations
    - param obs_std: Standard deviation of observations
    - param acs_mean: Mean of actions
    - param acs_std: Standard deviation of actions
    - param delta_mean: Mean of state difference `s_t+1 - s_t`.
    - param delta_std: Standard deviation of state difference `s_t+1 - s_t`
- the input for `delta_network` also should be normalized
#### `prediction`
- get the prediction from forward function.

#### `update`
- during training process, call `models.update()`
- the neural network for each `ff_models` is `delta_network`, built by `build_mlp`, 

### `MPC_policy`
#### `calculate_sum_of_rewards`
- `get_reward`
    - the input observation shape should be `(batch_size, ob_dim)` or `(ob_dim, )`, the action shape should be `(batchsize, ac_dim)` or `(ac_dim, )`
    - the output is a tuple with `r_total` and `terminal`
# Experiments
## 